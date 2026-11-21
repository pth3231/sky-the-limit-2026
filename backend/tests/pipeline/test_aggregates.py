from collections.abc import Callable
from typing import Literal

import pytest

from app.pipeline.aggregates import blend_recent, compute_aggregates
from app.pipeline.dataset import GameDraft
from app.pipeline.drafts import DraftAction, PlayerLine


def _action(
    sequence: int,
    kind: Literal["ban", "pick"],
    team_id: str,
    champion_id: str,
) -> DraftAction:
    return DraftAction(
        sequence_number=sequence,
        action_type=kind,
        team_id=team_id,
        champion_id=champion_id,
        champion_name=champion_id,
    )


def _game(
    game_id: str,
    patch: str,
    winner: str | None,
    actions: list[DraftAction],
    finished: bool = True,
) -> GameDraft:
    return GameDraft(
        game_id=game_id,
        series_id=f"series-{game_id}",
        game_number=1,
        patch=patch,
        finished=finished,
        winner_team_id=winner,
        blue_team_id="blue",
        red_team_id="red",
        actions=actions,
    )


def test_computes_champion_side_synergy_and_team_history_counts() -> None:
    games = [
        _game(
            "game-1",
            "15.10",
            "blue",
            [
                _action(1, "pick", "blue", "A"),
                _action(2, "pick", "blue", "B"),
                _action(3, "pick", "red", "C"),
                _action(4, "ban", "red", "X"),
            ],
        ),
        _game(
            "game-2",
            "15.10",
            "blue",
            [
                _action(1, "pick", "blue", "C"),
                _action(2, "pick", "red", "B"),
                _action(3, "ban", "red", "A"),
            ],
            finished=False,
        ),
        _game(
            "game-no-winner",
            "15.10",
            None,
            [_action(1, "pick", "red", "A")],
        ),
        _game(
            "game-3",
            "15.9",
            "red",
            [
                _action(1, "pick", "blue", "B"),
                _action(2, "pick", "blue", "A"),
                _action(3, "pick", "red", "C"),
            ],
        ),
    ]

    result = compute_aggregates(games)

    patch = result["15.10"]
    champions = {item.champion_id: item for item in patch.champions}
    assert patch.game_count == 3
    assert champions["A"].games == 2
    assert champions["A"].decided_games == 1
    assert champions["A"].picks == 2
    assert champions["A"].bans == 1
    assert champions["A"].wins == 1
    assert patch.sides.blue_games == 1
    assert patch.sides.blue_wins == 1
    assert patch.sides.red_games == 1
    assert patch.sides.red_wins == 0
    assert patch.synergies[0].champion_a_id == "A"
    assert patch.synergies[0].champion_b_id == "B"
    assert patch.synergies[0].games == 1
    assert patch.synergies[0].wins == 1

    team_a = next(
        item for item in patch.team_champions if item.team_id == "blue" and item.champion_id == "A"
    )
    assert team_a.picks == 1
    assert team_a.decided_picks == 1
    assert team_a.wins == 1
    assert team_a.banned_against == 1

    red_a = next(
        item for item in patch.team_champions if item.team_id == "red" and item.champion_id == "A"
    )
    assert red_a.picks == 1
    assert red_a.decided_picks == 0


def test_undecided_games_are_excluded_from_win_rate_denominators() -> None:
    games = [
        _game("decided", "15.10", "red", [_action(1, "pick", "blue", "A")]),
        _game(
            "unfinished",
            "15.10",
            "blue",
            [_action(1, "pick", "blue", "A"), _action(2, "pick", "blue", "B")],
            finished=False,
        ),
        _game("no-winner", "15.10", None, [_action(1, "pick", "red", "A")]),
    ]

    patch = compute_aggregates(games)["15.10"]
    champion_a = next(item for item in patch.champions if item.champion_id == "A")

    assert patch.game_count == 3
    assert patch.sides.blue_games == 1
    assert patch.sides.red_games == 1
    assert patch.sides.red_wins == 1
    assert champion_a.picks == 3
    assert champion_a.decided_games == 1
    assert champion_a.wins == 0
    assert patch.synergies == []


def test_counts_games_and_decided_wins_per_team() -> None:
    games = [
        _game("won", "15.10", "blue", []),
        _game("lost", "15.10", "red", []),
        _game("unfinished", "15.10", "blue", [], finished=False),
    ]

    patch = compute_aggregates(games)["15.10"]
    teams = {item.team_id: item for item in patch.teams}

    assert (teams["blue"].games, teams["blue"].decided_games, teams["blue"].wins) == (3, 2, 1)
    assert (teams["red"].games, teams["red"].decided_games, teams["red"].wins) == (3, 2, 1)


def test_blue_ban_is_recorded_against_red_team() -> None:
    game = _game("game-1", "15.10", "blue", [_action(1, "ban", "blue", "Z")])

    patch = compute_aggregates([game])["15.10"]
    history = {(item.team_id, item.champion_id): item for item in patch.team_champions}

    assert history[("blue", "Z")].bans == 1
    assert history[("blue", "Z")].banned_against == 0
    assert history[("red", "Z")].banned_against == 1


def test_action_from_team_outside_the_game_is_rejected() -> None:
    game = _game("game-1", "15.10", "blue", [_action(1, "pick", "stranger", "A")])

    with pytest.raises(ValueError, match="unknown team"):
        compute_aggregates([game])


def test_non_numeric_patch_is_labeled_unknown() -> None:
    odd_game = _game("odd-patch", "V15.1b", "blue", [])
    versioned_game = _game("versioned-patch", "15.9", "blue", [])

    aggregates = compute_aggregates([odd_game, versioned_game])
    blended = blend_recent({"V15.1b": [odd_game], "15.9": [versioned_game]}, min_games=2)

    assert list(aggregates) == ["unknown", "15.9"]
    assert [game.game_id for game in blended] == ["versioned-patch", "odd-patch"]


def test_blend_recent_rejects_non_positive_minimum() -> None:
    with pytest.raises(ValueError, match="min_games"):
        blend_recent({}, min_games=0)


def test_blends_newest_numeric_patch_versions_until_minimum_games() -> None:
    patches = {
        "15.9": [_game("older", "15.9", "blue", [])],
        "15.10": [
            _game("newest-1", "15.10", "blue", []),
            _game("newest-2", "15.10", "red", []),
        ],
        "15.8": [_game("oldest", "15.8", "blue", [])],
    }

    newest_only = blend_recent(patches, min_games=2)
    blended = blend_recent(patches, min_games=3)
    all_available = blend_recent(patches, min_games=10)

    assert [game.game_id for game in newest_only] == ["newest-1", "newest-2"]
    assert [game.game_id for game in blended] == ["newest-1", "newest-2", "older"]
    assert [game.game_id for game in all_available] == [
        "newest-1",
        "newest-2",
        "older",
        "oldest",
    ]


def test_computes_per_role_matchup_counts_from_lineups(make_game: Callable[..., GameDraft]) -> None:
    def lineup(team: str, role: str, champion: str) -> PlayerLine:
        return PlayerLine(team_id=team, role=role, champion_id=champion)

    blue_win = make_game(
        blue="blue", red="red", winner="blue", blue_picks=["A"], red_picks=["C"]
    ).model_copy(update={"lineups": [lineup("blue", "top", "A"), lineup("red", "top", "C")]})
    red_win = make_game(
        blue="blue", red="red", winner="red", blue_picks=["A"], red_picks=["C"]
    ).model_copy(update={"lineups": [lineup("blue", "top", "A"), lineup("red", "top", "C")]})
    missing_role = make_game(blue="blue", red="red", winner="blue", blue_picks=["A"]).model_copy(
        update={"lineups": [lineup("blue", "top", "A")]}
    )

    aggregates = compute_aggregates([blue_win, red_win, missing_role])
    matchups = {
        (item.role, item.champion_id, item.opponent_champion_id): item
        for item in aggregates["16.1"].matchups
    }

    assert (matchups["top", "A", "C"].games, matchups["top", "A", "C"].wins) == (2, 1)
    assert (matchups["top", "C", "A"].games, matchups["top", "C", "A"].wins) == (2, 1)


def test_blank_patch_is_labeled_unknown_and_used_after_numeric_patches() -> None:
    unknown_game = _game("unknown-patch", "", "blue", [])
    versioned_game = _game("versioned-patch", "15.9", "blue", [])

    aggregates = compute_aggregates([unknown_game, versioned_game])
    blended = blend_recent(
        {"unknown": [unknown_game], "15.9": [versioned_game]},
        min_games=2,
    )

    assert list(aggregates) == ["unknown", "15.9"]
    assert [game.game_id for game in blended] == ["versioned-patch", "unknown-patch"]
