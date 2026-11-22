from collections.abc import Mapping

from app.engine.draft_order import DRAFT_ORDER
from app.pipeline.meta import Champion
from app.pipeline.sources.oracles_elixir import (
    OE_POSITION_ROLES,
    _number,
    build_champion_id_map,
    build_game,
    resolve_champion_id,
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
    assert {line.role for line in game.lineups} == {"top", "jungle", "mid", "bot", "support"}


def test_build_game_skips_incomplete_draft() -> None:
    rows = _complete_game()
    rows = [row for row in rows if not (row["position"] == "sup" and row["side"] == "Blue")]
    assert build_game(rows, build_champion_id_map({})) is None
