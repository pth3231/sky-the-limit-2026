import csv
import json
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from app.engine.draft_order import DRAFT_ORDER
from app.pipeline.dataset import GameDraft
from app.pipeline.meta import Champion
from app.pipeline.sources.oracles_elixir import (
    OE_POSITION_ROLES,
    _number,
    build_champion_id_map,
    build_game,
    count_unmatched_champions,
    load_oracles_elixir,
    main,
    parse_oracles_elixir,
    resolve_champion_id,
)

DRAFT_ORDER_PICK_INDEX = next(
    i for i, slot in enumerate(DRAFT_ORDER) if slot.action == "pick" and slot.side == "blue"
)


def _champions() -> Mapping[str, Champion]:
    return {
        "Aatrox": Champion(character_id="grid-aatrox", name="Aatrox", games_played=1),
        "LeeSin": Champion(character_id="grid-leesin", name="Lee Sin", games_played=1),
    }


def test_position_roles_cover_oe_abbreviations() -> None:
    assert OE_POSITION_ROLES == {
        "top": "top",
        "jng": "jungle",
        "mid": "mid",
        "bot": "bot",
        "sup": "support",
    }


def test_resolve_champion_id_maps_names_to_grid_ids_with_fallback() -> None:
    name_to_id = build_champion_id_map(_champions())
    assert resolve_champion_id("Aatrox", name_to_id) == "grid-aatrox"
    # Normalization ignores spaces/case, so OE "Lee Sin" matches GRID "Lee Sin".
    assert resolve_champion_id("Lee Sin", name_to_id) == "grid-leesin"
    # Unknown champion falls back to its normalized key, never raising.
    assert resolve_champion_id("Smolder", name_to_id) == "smolder"


def test_number_coerces_blanks_to_zero() -> None:
    assert _number("12.5") == 12.5
    assert _number("") == 0.0
    assert _number(None) == 0.0
    assert _number("n/a") == 0.0


def _player_row(gid, side, pos, champ, **stats):
    row = {
        "gameid": gid,
        "datacompleteness": "complete",
        "date": "2026-11-01 10:00:00",
        "patch": "16.20",
        "side": side,
        "position": pos,
        "teamname": f"{side} Team",
        "teamid": f"oe-{side.lower()}",
        "champion": champ,
        "playerid": f"p-{side}-{pos}",
        "kills": "0",
        "assists": "0",
        "teamkills": "10",
        "damageshare": "0.2",
        "totaldamagetaken": "15000",
        "visionscore": "30",
        "earnedgoldshare": "0.2",
        "result": "1" if side == "Blue" else "0",
    }
    row.update({key: str(value) for key, value in stats.items()})
    return row


def _team_row(gid, side, bans, result):
    row = {
        "gameid": gid,
        "datacompleteness": "complete",
        "date": "2026-11-01 10:00:00",
        "patch": "16.20",
        "side": side,
        "position": "team",
        "teamname": f"{side} Team",
        "teamid": f"oe-{side.lower()}",
        "champion": "",
        "result": result,
        "kills": "10",
        "assists": "0",
        "teamkills": "10",
        "damageshare": "0",
        "totaldamagetaken": "0",
        "visionscore": "0",
        "earnedgoldshare": "0",
    }
    for index, ban in enumerate(bans, start=1):
        row[f"ban{index}"] = ban
    return row


def _complete_game(gid="g1"):
    positions = ("top", "jng", "mid", "bot", "sup")
    rows = []
    for side in ("Blue", "Red"):
        for pos in positions:
            rows.append(_player_row(gid, side, pos, f"{side}{pos}"))
    rows.append(_team_row(gid, "Blue", ["Bb1", "Bb2", "Bb3", "Bb4", "Bb5"], "1"))
    rows.append(_team_row(gid, "Red", ["Rb1", "Rb2", "Rb3", "Rb4", "Rb5"], "0"))
    return rows


def test_build_game_emits_full_draft_order_aligned_sequence() -> None:
    game = build_game(_complete_game(), build_champion_id_map({}))

    assert game is not None
    assert game.winner_team_id == game.blue_team_id == "oe-blue"
    assert game.red_team_id == "oe-red"
    assert len(game.actions) == len(DRAFT_ORDER)
    for slot, action in zip(DRAFT_ORDER, game.actions, strict=True):
        assert action.action_type == slot.action
        expected_team = game.blue_team_id if slot.side == "blue" else game.red_team_id
        assert action.team_id == expected_team
    assert len(game.lineups) == 10
    assert {line.player_id for line in game.lineups} >= {"p-Blue-top", "p-Red-sup"}
    assert {line.role for line in game.lineups} == {"top", "jungle", "mid", "bot", "support"}


def test_build_game_skips_incomplete_draft() -> None:
    rows = _complete_game()
    rows = [row for row in rows if not (row["position"] == "sup" and row["side"] == "Blue")]
    assert build_game(rows, build_champion_id_map({})) is None


def test_parse_keeps_complete_games_and_drops_incomplete() -> None:
    good = _complete_game("good")
    partial = [row for row in _complete_game("partial") if row["position"] != "sup"]
    flagged = _complete_game("flagged")
    for row in flagged:
        row["datacompleteness"] = "partial"

    games = parse_oracles_elixir([*good, *partial, *flagged], {})

    assert [game.game_id for game in games] == ["oe:good"]


def test_load_reads_csv_and_resolves_champions(tmp_path: Path) -> None:
    rows = _complete_game("g1")
    csv_path = tmp_path / "oe.csv"
    fieldnames = sorted({key for row in rows for key in row})
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    champions = {"Bluetop": Champion(character_id="grid-x", name="Bluetop", games_played=1)}

    games = load_oracles_elixir(csv_path, champions)

    assert len(games) == 1
    assert games[0].actions[DRAFT_ORDER_PICK_INDEX].champion_id == "grid-x"


def test_main_appends_oe_games_to_existing_drafts(tmp_path, capsys, monkeypatch) -> None:
    rows = _complete_game("g1")
    csv_path = tmp_path / "oe.csv"
    fieldnames = sorted({key for row in rows for key in row})
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    out = tmp_path / "derived"
    out.mkdir()
    (out / "drafts.json").write_text("[]", encoding="utf-8")

    monkeypatch.setattr("app.pipeline.sources.oracles_elixir.load_champions", lambda _data_dir: {})
    main(["--csv", str(csv_path), "--data-dir", str(tmp_path), "--out", str(out)])

    payload = json.loads((out / "drafts.json").read_text(encoding="utf-8"))
    games = [GameDraft.model_validate(item) for item in payload]
    assert [game.game_id for game in games] == ["oe:g1"]
    assert "unmatched champion" in capsys.readouterr().out.lower()


def test_count_unmatched_champions_counts_distinct_names_without_grid_match() -> None:
    champions = {"Aatrox": Champion(character_id="grid-aatrox", name="Aatrox", games_played=1)}
    name_to_id = build_champion_id_map(champions)
    games = [
        build_game(_complete_game("g1"), name_to_id),
        build_game(_complete_game("g2"), name_to_id),
    ]
    assert all(game is not None for game in games)

    unmatched = count_unmatched_champions(games, name_to_id)  # type: ignore[arg-type]

    # Every champion/ban name in _complete_game's synthetic rows is unmatched against
    # the single known champion "Aatrox", so the distinct unmatched count is positive
    # and stable across games that reuse the same synthetic names.
    assert unmatched > 0


def test_module_runs_main_when_invoked_as_a_script(tmp_path: Path, monkeypatch: Any) -> None:
    """Regression test: `python -m app.pipeline.sources.oracles_elixir` must invoke main().

    A missing `if __name__ == "__main__":` guard makes the CLI silently do nothing when
    run via `-m` (exit 0, no output, no file write) while every direct call to `main()`
    still works — easy to miss in review since it only breaks the command-line entrypoint.
    """
    rows = _complete_game("g1")
    csv_path = tmp_path / "oe.csv"
    fieldnames = sorted({key for row in rows for key in row})
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    data_dir = tmp_path / "data"
    (data_dir / "meta").mkdir(parents=True)
    (data_dir / "meta" / "characters.csv").write_text(
        "title,character_id,name,games_played\n", encoding="utf-8"
    )
    out = tmp_path / "derived"
    out.mkdir()
    (out / "drafts.json").write_text("[]", encoding="utf-8")

    monkeypatch.chdir(Path(__file__).resolve().parents[2])
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "app.pipeline.sources.oracles_elixir",
            "--csv",
            str(csv_path),
            "--data-dir",
            str(data_dir),
            "--out",
            str(out),
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "unmatched champion" in result.stdout.lower()
    payload = json.loads((out / "drafts.json").read_text(encoding="utf-8"))
    assert [item["game_id"] for item in payload] == ["oe:g1"]
