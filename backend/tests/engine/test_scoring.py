import math
from collections.abc import Callable

import pytest

from app.engine.scoring import Scorer, series_win_probability, signal_for, smoothed_rate
from app.engine.stats import blend_patches
from app.pipeline.aggregates import compute_aggregates
from app.pipeline.dataset import GameDraft
from app.pipeline.drafts import PlayerLine

NAMES = {"A": "Ahri", "W": "Weakling", "L": "Lee Sin", "O": "Orianna", "N": "Nami", "R": "Rare"}


@pytest.fixture
def scorer(make_game: Callable[..., GameDraft]) -> Scorer:
    def games(times: int, **kwargs: object) -> list[GameDraft]:
        return [make_game(blue="t1", red="t2", **kwargs) for _ in range(times)]

    drafts = [
        *games(30, winner="blue", blue_picks=["A"]),
        *games(10, winner="red", blue_picks=["A"]),
        *games(10, winner="blue", blue_picks=["W"]),
        *games(30, winner="red", blue_picks=["W"]),
        *[make_game(blue="c9", red="t2", winner="blue", blue_picks=["A", "L"]) for _ in range(9)],
        make_game(blue="c9", red="t2", winner="red", blue_picks=["A", "L"]),
        *games(10, winner="red", blue_picks=["L"]),
        *[make_game(red="opp", winner="red", red_picks=["O"]) for _ in range(8)],
        *[make_game(red="opp", winner="blue", red_picks=["O"]) for _ in range(2)],
        *[make_game(blue="t1", red="t2", winner="blue", blue_picks=["R"]) for _ in range(3)],
    ]
    stats = blend_patches(compute_aggregates(drafts), None, min_games=1)
    return Scorer(
        stats,
        champion_names=NAMES,
        our_team_id="c9",
        our_team_name="Cloud9",
        opponent_team_id="opp",
        opponent_team_name="Rivals",
        our_side="blue",
        champion_roles={},
    )


@pytest.fixture
def matchup_scorer(make_game: Callable[..., GameDraft]) -> Scorer:
    def lineup(team: str, role: str, champion: str) -> PlayerLine:
        return PlayerLine(team_id=team, role=role, champion_id=champion)

    def game(winner: str) -> GameDraft:
        return make_game(
            blue="c9", red="opp", winner=winner, blue_picks=["A"], red_picks=["X"]
        ).model_copy(update={"lineups": [lineup("c9", "top", "A"), lineup("opp", "top", "X")]})

    games = [game("blue") for _ in range(8)] + [game("red") for _ in range(2)]
    stats = blend_patches(compute_aggregates(games), None, min_games=1)
    return Scorer(
        stats,
        champion_names={"A": "Ahri", "X": "Xin Zhao"},
        our_team_id="c9",
        our_team_name="Cloud9",
        opponent_team_id="opp",
        opponent_team_name="Rivals",
        our_side="blue",
        champion_roles={"A": "top", "X": "top"},
    )


def test_matchup_factor_appears_when_opponent_has_same_role_pick(matchup_scorer: Scorer) -> None:
    candidate = matchup_scorer.score_pick("A", our_picks=(), opponent_picks=("X",))

    labels = [factor.label for factor in candidate.contributing_factors]
    assert any("beats their Xin Zhao" in label for label in labels)


def test_matchup_factor_absent_without_a_same_role_opposing_pick(matchup_scorer: Scorer) -> None:
    candidate = matchup_scorer.score_pick("A", our_picks=(), opponent_picks=())

    labels = [factor.label for factor in candidate.contributing_factors]
    assert not any("lane matchup" in label.lower() for label in labels)


def test_ban_candidate_surfaces_matchup_threat_against_our_pick(matchup_scorer: Scorer) -> None:
    candidate = matchup_scorer.score_ban("X", our_picks=("A",), opponent_picks=())

    labels = [factor.label for factor in candidate.contributing_factors]
    assert any("loses to your Ahri" in label for label in labels)


def test_smoothed_rate_pulls_small_samples_toward_even() -> None:
    assert smoothed_rate(0, 0) == 0.5
    assert smoothed_rate(10, 10) == 0.75
    assert smoothed_rate(300, 400) == pytest.approx(0.744, abs=0.001)


def test_signal_qualifiers_follow_sample_size() -> None:
    assert [signal_for(games) for games in (0, 9, 10, 29, 30)] == [
        "limited",
        "limited",
        "some",
        "some",
        "strong",
    ]


def test_strong_champion_outranks_weak_champion(scorer: Scorer) -> None:
    strong = scorer.score_pick("A", our_picks=(), opponent_picks=())
    weak = scorer.score_pick("W", our_picks=(), opponent_picks=())

    assert strong.score > scorer.game_win_probability((), ()) > weak.score
    assert strong.contributing_factors[0].label.startswith("Wins 78% of pro games")
    assert strong.contributing_factors[0].signal == "strong"
    assert strong.reason.startswith("Raises your estimated game win chance to")


def test_synergy_factor_names_the_teammate(scorer: Scorer) -> None:
    candidate = scorer.score_pick("A", our_picks=("L",), opponent_picks=())

    labels = [factor.label for factor in candidate.contributing_factors]
    assert "Strong with your Lee Sin (won 9 of 10 together)" in labels
    assert "Cloud9 won 9 of 10 games with it" in labels


def test_small_samples_are_marked_limited(scorer: Scorer) -> None:
    candidate = scorer.score_pick("R", our_picks=(), opponent_picks=())
    unseen = scorer.score_pick("N", our_picks=(), opponent_picks=())

    assert candidate.contributing_factors[0].signal == "limited"
    assert unseen.contributing_factors[0].label == "No recent pro games on these patches"
    assert unseen.contributing_factors[0].signal == "limited"


def test_pick_reason_distinguishes_raise_lower_and_unchanged(scorer: Scorer) -> None:
    unchanged = scorer.score_pick("N", our_picks=(), opponent_picks=())
    lowered = scorer.score_pick("W", our_picks=(), opponent_picks=())

    assert unchanged.reason.startswith("Keeps your estimated game win chance unchanged at")
    assert lowered.reason.startswith("Lowers your estimated game win chance to")


def test_factors_are_ordered_by_impact(scorer: Scorer) -> None:
    candidate = scorer.score_pick("A", our_picks=("L",), opponent_picks=())

    impacts = [abs(factor.impact) for factor in candidate.contributing_factors]
    assert impacts == sorted(impacts, reverse=True)


def test_factor_impacts_use_probability_points(scorer: Scorer) -> None:
    candidate = scorer.score_pick("A", our_picks=(), opponent_picks=())
    meta = next(factor for factor in candidate.contributing_factors if "pro games" in factor.label)
    expected = round(1 / (1 + math.exp(-1.2 * 0.1)) - 0.5, 3)

    assert meta.impact == expected


def test_ban_targets_the_opponents_comfort_pick(scorer: Scorer) -> None:
    comfort_ban = scorer.score_ban("O", our_picks=(), opponent_picks=())
    idle_ban = scorer.score_ban("N", our_picks=(), opponent_picks=())

    labels = [factor.label for factor in comfort_ban.contributing_factors]
    assert comfort_ban.score > idle_ban.score
    assert "Rivals picked it in 10 of 10 games" in labels
    assert comfort_ban.reason.startswith("Takes away")


def test_game_win_probability_moves_with_compositions(scorer: Scorer) -> None:
    even = scorer.game_win_probability((), ())

    assert scorer.game_win_probability(("A",), ()) > even
    assert scorer.game_win_probability((), ("A",)) < even
    assert scorer.composition_factors(("A",), ("W",))


def test_stacked_signals_stay_believable(make_game: Callable[..., GameDraft]) -> None:
    lineup = ["A", "L", "O", "N", "R"]
    drafts = [make_game(winner="blue", blue_picks=lineup) for _ in range(40)]
    stats = blend_patches(compute_aggregates(drafts), None, min_games=1)
    scorer = Scorer(
        stats,
        champion_names=NAMES,
        our_team_id="c9",
        our_team_name="Cloud9",
        opponent_team_id="opp",
        opponent_team_name="Rivals",
        our_side="blue",
        champion_roles={},
    )

    assert 0.7 < scorer.game_win_probability(lineup, ()) < 0.85


def test_side_factor_reports_no_decided_game_evidence(
    make_game: Callable[..., GameDraft],
) -> None:
    stats = blend_patches(compute_aggregates([make_game(winner=None)]), None, min_games=1)
    scorer = Scorer(
        stats,
        champion_names=NAMES,
        our_team_id="c9",
        our_team_name="Cloud9",
        opponent_team_id="opp",
        opponent_team_name="Rivals",
        our_side="blue",
        champion_roles={},
    )

    assert scorer.composition_factors((), ())[0].label == "No decided pro games on these patches"


@pytest.mark.parametrize(
    ("probability", "score", "series_format", "expected"),
    [
        (0.6, (0, 0), "bo1", 0.6),
        (0.5, (1, 0), "bo3", 0.75),
        (1.0, (0, 2), "bo5", 0.25),
        (0.5, (0, 0), "bo3", 0.5),
    ],
)
def test_series_win_probability(
    probability: float,
    score: tuple[int, int],
    series_format: str,
    expected: float,
) -> None:
    result = series_win_probability(probability, *score, series_format)  # type: ignore[arg-type]

    assert result == pytest.approx(expected)
