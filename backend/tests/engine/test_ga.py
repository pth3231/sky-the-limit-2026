from collections.abc import Sequence
from typing import Any

from app.engine.fearless import build_context
from app.engine.ga import search_moves
from app.engine.models import DraftRequest, PreviousGame
from app.engine.stats import blend_patches
from app.model.features import DraftState
from app.pipeline.aggregates import compute_aggregates


class StubPredictor:
    def __init__(self) -> None:
        self.seen: list[DraftState] = []

    def predict(self, states: Sequence[DraftState]) -> list[float]:
        self.seen.extend(states)
        return [0.9 if state.actions[6] == "best" else 0.1 for state in states]


def _request() -> DraftRequest:
    return DraftRequest(
        opponent_team_id="opp",
        patch="16.1",
        our_side="blue",
        previous_games=[
            PreviousGame(
                our_picks=["fearless", "old-our-1", "old-our-2", "old-our-3", "old-our-4"],
                opponent_picks=["prior", "old-opp-1", "old-opp-2", "old-opp-3", "old-opp-4"],
                result="loss",
            )
        ],
        actions=[f"ban-{index}" for index in range(6)],
    )


def test_exhaustive_search_ranks_distinct_legal_moves_and_uses_opponent_history(
    make_game: Any,
) -> None:
    request = _request()
    candidates = ["best", "reply", "candidate-1", "candidate-2", "candidate-3", "candidate-4"]
    known = {
        *candidates,
        "fearless",
        "prior",
        *(f"old-our-{index}" for index in range(1, 5)),
        *(f"old-opp-{index}" for index in range(1, 5)),
        *(f"ban-{index}" for index in range(6)),
    }
    context = build_context(request, known)
    history = [make_game(red_picks=["reply"]) for _ in range(3)]
    stats = blend_patches(compute_aggregates(history), patch="16.1", min_games=1)
    predictor = StubPredictor()

    ranked = search_moves(
        request,
        context,
        stats,
        "opp",
        candidates,
        predictor,
        result_count=4,
        seed=7,
        exhaustive_threshold=256,
    )

    assert ranked[0][0] == "best"
    assert len({champion_id for champion_id, _ in ranked}) == len(ranked)
    assert all(champion_id not in context.unavailable for champion_id, _ in ranked)
    best_first_states = [state for state in predictor.seen if state.actions[6] == "best"]
    assert best_first_states
    assert all(state.actions[7] == "reply" for state in best_first_states)
    assert all(len(state.actions) == len(set(state.actions)) for state in predictor.seen)
    assert all("fearless" not in state.actions for state in predictor.seen)


def test_genetic_search_is_reproducible(make_game: Any) -> None:
    request = _request()
    candidates = ["best", "reply", "candidate-1", "candidate-2", "candidate-3", "candidate-4"]
    known = {
        *candidates,
        "fearless",
        "prior",
        *(f"old-our-{index}" for index in range(1, 5)),
        *(f"old-opp-{index}" for index in range(1, 5)),
        *(f"ban-{index}" for index in range(6)),
    }
    context = build_context(request, known)
    stats = blend_patches(
        compute_aggregates([make_game(red_picks=["reply"])]),
        patch="16.1",
        min_games=1,
    )

    first = search_moves(
        request,
        context,
        stats,
        "opp",
        candidates,
        StubPredictor(),
        result_count=4,
        seed=31,
        population_size=12,
        generations=4,
        exhaustive_threshold=0,
    )
    second = search_moves(
        request,
        context,
        stats,
        "opp",
        candidates,
        StubPredictor(),
        result_count=4,
        seed=31,
        population_size=12,
        generations=4,
        exhaustive_threshold=0,
    )

    assert first == second
