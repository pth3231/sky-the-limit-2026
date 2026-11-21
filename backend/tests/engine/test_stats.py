from collections.abc import Callable

import pytest

from app.engine.errors import UnknownPatchError
from app.engine.stats import blend_patches
from app.pipeline.aggregates import PatchAggregate, compute_aggregates
from app.pipeline.dataset import GameDraft


@pytest.fixture
def aggregates(make_game: Callable[..., GameDraft]) -> dict[str, PatchAggregate]:
    games = [
        *(make_game("16.8", blue_picks=["A"]) for _ in range(5)),
        *(make_game("16.9", blue_picks=["A", "B"], red_bans=["C"]) for _ in range(5)),
        *(
            make_game("16.10", winner="red", blue_picks=["A", "B"], red_picks=["C"])
            for _ in range(3)
        ),
    ]
    return compute_aggregates(games)


def test_blends_newest_numeric_patches_until_the_minimum(
    aggregates: dict[str, PatchAggregate],
) -> None:
    stats = blend_patches(aggregates, None, min_games=6)

    assert stats.patches == ("16.10", "16.9")
    assert stats.game_count == 8


def test_requested_older_patch_excludes_newer_patches(
    aggregates: dict[str, PatchAggregate],
) -> None:
    stats = blend_patches(aggregates, "16.9", min_games=1)

    assert stats.patches == ("16.9",)
    assert stats.game_count == 5


def test_uses_every_older_patch_when_the_minimum_is_not_reached(
    aggregates: dict[str, PatchAggregate],
) -> None:
    stats = blend_patches(aggregates, None, min_games=100)

    assert stats.patches == ("16.10", "16.9", "16.8")


def test_sums_counts_across_blended_patches(aggregates: dict[str, PatchAggregate]) -> None:
    stats = blend_patches(aggregates, None, min_games=6)

    champion = stats.champion("A")
    pair = stats.pair("B", "A")
    comfort = stats.team_champion("c9", "A")
    assert champion is not None and pair is not None and comfort is not None
    assert (champion.picks, champion.decided_games, champion.wins) == (13, 13, 10)
    assert (pair.games, pair.wins) == (8, 5)
    assert (comfort.picks, comfort.wins) == (13, 10)
    assert stats.champion("C") is not None
    assert stats.team_champion("opp", "C") is not None
    assert stats.side_record("blue") == (5, 8)
    assert stats.side_record("red") == (3, 8)
    assert stats.team_games("c9") == 13
    assert stats.champion("missing") is None
    assert stats.pair("A", "missing") is None


def test_sums_matchup_counts_across_blended_patches(aggregates: dict[str, PatchAggregate]) -> None:
    stats = blend_patches(aggregates, None, min_games=6)

    assert stats.matchup("top", "missing", "missing") is None


def test_unknown_or_missing_patch_raises(aggregates: dict[str, PatchAggregate]) -> None:
    with pytest.raises(UnknownPatchError, match="15.1"):
        blend_patches(aggregates, "15.1")
    with pytest.raises(UnknownPatchError):
        blend_patches({}, None)


def test_subject_history_extends_beyond_global_patch_window(
    make_game: Callable[..., GameDraft],
) -> None:
    games = [
        *(make_game("16.2", blue="t1", red="t2", blue_picks=["A"]) for _ in range(60)),
        *(
            make_game("16.1", blue="c9", red="opp", blue_picks=["X"], red_picks=["S"])
            for _ in range(10)
        ),
    ]

    stats = blend_patches(compute_aggregates(games), None)

    assert stats.patches == ("16.2",)
    assert stats.game_count == 60
    assert stats.team_games("c9") == 10
    assert stats.team_games("opp") == 10
    assert stats.team_patches("opp") == ("16.1",)
    assert stats.all_patches() == ("16.2", "16.1")
    team_champion = stats.team_champion("c9", "X")
    champion = stats.champion("S")
    assert team_champion is not None and team_champion.picks == 10
    assert champion is not None and champion.games == 10
