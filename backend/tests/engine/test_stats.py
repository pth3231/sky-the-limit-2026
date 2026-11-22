import time
from collections.abc import Callable

import pytest

from app.engine.errors import UnknownPatchError
from app.engine.stats import blend_patches
from app.pipeline.aggregates import PatchAggregate, SideAggregate, TeamAggregate, compute_aggregates
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


def test_blend_patches_scales_with_total_records_not_patches_times_subjects() -> None:
    """Regression test: per-subject lookback must not linearly scan each patch's team list.

    Real multi-year tournament data has many teams that each appear on only a handful of
    patches, so most (team, patch) pairs are misses. Scanning every patch's full team list
    with `next(... for item in patch.teams if item.team_id == team_id)` to resolve each miss
    made this scale with teams_per_patch * patch_count * teams_per_patch, which on the merged
    GRID + Oracle's Elixir dataset (83 patches, 927 teams) made validation-metric computation
    take minutes instead of a fraction of a second.
    """
    patch_count = 83
    teams_per_patch = 300
    aggregates: dict[str, PatchAggregate] = {}
    for patch_index in range(patch_count):
        label = f"20.{patch_index:02d}"
        teams = [
            TeamAggregate(
                team_id=f"team-{patch_index}-{team_index}", games=20, decided_games=20, wins=10
            )
            for team_index in range(teams_per_patch)
        ]
        aggregates[label] = PatchAggregate(
            patch=label,
            game_count=teams_per_patch,
            champions=[],
            sides=SideAggregate(blue_games=0, blue_wins=0, red_games=0, red_wins=0),
            synergies=[],
            team_champions=[],
            teams=teams,
        )

    started = time.perf_counter()
    blend_patches(aggregates, patch=f"20.{patch_count - 1:02d}", min_games=1)
    elapsed = time.perf_counter() - started

    assert elapsed < 2.0
