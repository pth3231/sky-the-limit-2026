"""Parse Oracle's Elixir match CSVs into the pipeline's GameDraft shape."""

import argparse
import csv
import json
from collections import OrderedDict
from collections.abc import Iterable, Mapping
from pathlib import Path

from app.engine.draft_order import DRAFT_ORDER
from app.pipeline.dataset import GameDraft
from app.pipeline.ddragon import normalize_champion_name
from app.pipeline.drafts import DraftAction, PlayerLine, Role
from app.pipeline.meta import Champion, load_champions

OE_POSITION_ROLES: dict[str, Role] = {
    "top": "top",
    "jng": "jungle",
    "mid": "mid",
    "bot": "bot",
    "sup": "support",
}


def _number(value: object) -> float:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0.0


def build_champion_id_map(champions: Mapping[str, Champion]) -> dict[str, str]:
    """Map normalized champion display names to GRID character ids."""
    return {
        normalize_champion_name(champion.name): champion.character_id
        for champion in champions.values()
    }


def resolve_champion_id(name: str, name_to_id: Mapping[str, str]) -> str:
    """Resolve an OE champion name to a GRID id, falling back to its normalized key."""
    normalized = normalize_champion_name(name)
    return name_to_id.get(normalized, normalized)


def count_unmatched_champions(games: list[GameDraft], name_to_id: Mapping[str, str]) -> int:
    """Count distinct champion names in `games` with no entry in the GRID id map.

    These champions fell back to their normalized name as an id in `resolve_champion_id`,
    so they don't share an embedding slot with any GRID champion of the same name.
    """
    unmatched = {
        action.champion_name
        for game in games
        for action in game.actions
        if normalize_champion_name(action.champion_name) not in name_to_id
    }
    return len(unmatched)


_POSITION_ORDER = ("top", "jng", "mid", "bot", "sup")


def _team_id(row: Mapping[str, str]) -> str:
    return row.get("teamid", "").strip() or normalize_champion_name(row.get("teamname", ""))


def build_game(rows: list[Mapping[str, str]], name_to_id: Mapping[str, str]) -> GameDraft | None:
    """Build one OE game into a GameDraft, or None if its draft is incomplete."""
    team_rows = {
        row["side"].strip().capitalize(): row
        for row in rows
        if row.get("position", "").strip().lower() == "team"
    }
    if set(team_rows) != {"Blue", "Red"}:
        return None

    players: dict[str, list[Mapping[str, str]]] = {"Blue": [], "Red": []}
    for row in rows:
        position = row.get("position", "").strip().lower()
        side = row.get("side", "").strip().capitalize()
        if position in OE_POSITION_ROLES and side in players:
            players[side].append(row)

    picks: dict[str, list[Mapping[str, str]]] = {}
    bans: dict[str, list[str]] = {}
    for side in ("Blue", "Red"):
        ordered = sorted(
            players[side],
            key=lambda row: _POSITION_ORDER.index(row["position"].strip().lower()),
        )
        side_bans = [team_rows[side].get(f"ban{index}", "").strip() for index in range(1, 6)]
        if len(ordered) != 5 or any(not ban for ban in side_bans):
            return None
        picks[side] = ordered
        bans[side] = side_bans

    team_ids = {side: _team_id(team_rows[side]) for side in ("Blue", "Red")}
    winner_side = next(
        (side for side in ("Blue", "Red") if team_rows[side].get("result", "").strip() == "1"),
        None,
    )

    pick_cursor = {"Blue": 0, "Red": 0}
    ban_cursor = {"Blue": 0, "Red": 0}
    actions: list[DraftAction] = []
    for slot in DRAFT_ORDER:
        side = "Blue" if slot.side == "blue" else "Red"
        if slot.action == "ban":
            name = bans[side][ban_cursor[side]]
            ban_cursor[side] += 1
        else:
            name = picks[side][pick_cursor[side]]["champion"].strip()
            pick_cursor[side] += 1
        actions.append(
            DraftAction(
                sequence_number=slot.index + 1,
                action_type=slot.action,
                team_id=team_ids[side],
                champion_id=resolve_champion_id(name, name_to_id),
                champion_name=name,
            )
        )

    lineups: list[PlayerLine] = []
    for side in ("Blue", "Red"):
        for row in picks[side]:
            team_kills = _number(row.get("teamkills"))
            participations = _number(row.get("kills")) + _number(row.get("assists"))
            role: Role = OE_POSITION_ROLES[row["position"].strip().lower()]
            lineups.append(
                PlayerLine(
                    team_id=team_ids[side],
                    role=role,
                    champion_id=resolve_champion_id(row["champion"].strip(), name_to_id),
                    player_id=row.get("playerid", "").strip() or None,
                    damage_share=_number(row.get("damageshare")),
                    damage_taken=_number(row.get("totaldamagetaken")),
                    vision_score=_number(row.get("visionscore")),
                    gold_share=_number(row.get("earnedgoldshare")),
                    kill_participation=participations / team_kills if team_kills else 0.0,
                )
            )

    gameid = "oe:" + rows[0]["gameid"].strip()
    return GameDraft(
        game_id=gameid,
        series_id=gameid,
        game_number=1,
        patch=(team_rows["Blue"].get("patch", "").strip() or "unknown"),
        finished=True,
        winner_team_id=team_ids[winner_side] if winner_side else None,
        blue_team_id=team_ids["Blue"],
        red_team_id=team_ids["Red"],
        actions=actions,
        started_at=team_rows["Blue"].get("date", "").strip(),
        lineups=lineups,
    )


def parse_oracles_elixir(
    rows: Iterable[Mapping[str, str]], name_to_id: Mapping[str, str]
) -> list[GameDraft]:
    """Group OE rows by game and build every complete game into a GameDraft."""
    grouped: OrderedDict[str, list[Mapping[str, str]]] = OrderedDict()
    for row in rows:
        gameid = row.get("gameid", "").strip()
        if gameid:
            grouped.setdefault(gameid, []).append(row)

    games: list[GameDraft] = []
    for group in grouped.values():
        if any(row.get("datacompleteness", "").strip().lower() != "complete" for row in group):
            continue
        game = build_game(group, name_to_id)
        if game is not None:
            games.append(game)
    return games


def load_oracles_elixir(
    csv_path: Path,
    champions: Mapping[str, Champion],
) -> list[GameDraft]:
    """Read an Oracle's Elixir CSV into GameDrafts using GRID champion ids."""
    name_to_id = build_champion_id_map(champions)
    with csv_path.open(newline="", encoding="utf-8-sig") as file:
        return parse_oracles_elixir(list(csv.DictReader(file)), name_to_id)


def main(argv: list[str] | None = None) -> None:
    """Merge an Oracle's Elixir CSV into the local drafts.json training set."""
    parser = argparse.ArgumentParser(description="Add Oracle's Elixir games to drafts.json.")
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    arguments = parser.parse_args(argv)

    champions = load_champions(arguments.data_dir)
    name_to_id = build_champion_id_map(champions)
    oe_games = load_oracles_elixir(arguments.csv, champions)
    unmatched_count = count_unmatched_champions(oe_games, name_to_id)

    drafts_path = arguments.out / "drafts.json"
    existing: list[GameDraft] = []
    if drafts_path.is_file():
        existing = [
            GameDraft.model_validate(item)
            for item in json.loads(drafts_path.read_text(encoding="utf-8"))
        ]
    by_id = {game.game_id: game for game in existing}
    for game in oe_games:
        by_id[game.game_id] = game
    merged = list(by_id.values())
    drafts_path.write_text(
        json.dumps([game.model_dump(mode="json") for game in merged], indent=0) + "\n",
        encoding="utf-8",
    )
    print(
        f"Added {len(oe_games)} Oracle's Elixir games; drafts.json now has {len(merged)}. "
        f"{unmatched_count} unmatched champion name(s) fell back to a normalized id."
    )


if __name__ == "__main__":
    main()
