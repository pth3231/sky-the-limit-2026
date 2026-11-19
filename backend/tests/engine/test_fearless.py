from typing import Any

import pytest

from app.engine.errors import DraftStateError
from app.engine.fearless import build_context, fearless_excluded
from app.engine.models import DraftRequest, PreviousGame

CHAMPIONS = {f"c{number}" for number in range(40)}


def _previous(start: int, result: str = "win") -> dict[str, Any]:
    return {
        "ourPicks": [f"c{start + offset}" for offset in range(5)],
        "opponentPicks": [f"c{start + 5 + offset}" for offset in range(5)],
        "result": result,
    }


def _request(**overrides: Any) -> DraftRequest:
    payload: dict[str, Any] = {
        "opponentTeamId": "opp",
        "seriesFormat": "bo3",
        "ourSide": "blue",
        "previousGames": [],
        "actions": [],
    }
    payload.update(overrides)
    return DraftRequest.model_validate(payload)


def test_fearless_pool_shrinks_with_each_previous_game() -> None:
    game_one = _request()
    game_two = _request(previousGames=[_previous(0)])
    game_three = _request(previousGames=[_previous(0), _previous(10, "loss")])

    assert fearless_excluded(game_one.previous_games) == set()
    assert len(fearless_excluded(game_two.previous_games)) == 10
    assert len(fearless_excluded(game_three.previous_games)) == 20


def test_bans_do_not_carry_over_but_picks_do() -> None:
    previous = PreviousGame(our_picks=["c0"], opponent_picks=["c1"], result="win")

    assert fearless_excluded([previous]) == {"c0", "c1"}


def test_context_splits_actions_by_slot_and_reports_next_slot() -> None:
    actions = ["c20", "c21", "c22", "c23", "c24", "c25", "c26"]

    context = build_context(_request(ourSide="red", actions=actions), CHAMPIONS)

    assert context.game_number == 1
    assert context.our_bans == ("c21", "c23", "c25")
    assert context.opponent_bans == ("c20", "c22", "c24")
    assert context.opponent_picks == ("c26",)
    assert context.our_picks == ()
    assert context.next_slot is not None
    assert (context.next_slot.index, context.next_slot.side) == (7, "red")
    assert context.unavailable == frozenset(actions)


def test_context_marks_fearless_champions_unavailable() -> None:
    context = build_context(_request(previousGames=[_previous(0)]), CHAMPIONS)

    assert context.game_number == 2
    assert context.series_score == (1, 0)
    assert context.fearless_excluded == frozenset(f"c{number}" for number in range(10))
    assert context.fearless_excluded <= context.unavailable


def test_complete_draft_has_no_next_slot() -> None:
    context = build_context(_request(actions=[f"c{number}" for number in range(20)]), CHAMPIONS)

    assert context.next_slot is None
    assert len(context.our_picks) == len(context.opponent_picks) == 5


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"actions": ["c0", "c0"]}, "more than once"),
        ({"actions": ["missing"]}, "Unknown champion"),
        ({"actions": [f"c{number}" for number in range(21)]}, "at most 20"),
        ({"previousGames": [_previous(0)], "actions": ["c3"]}, "Fearless"),
        ({"previousGames": [_previous(0), _previous(3)]}, "Fearless"),
        (
            {
                "previousGames": [
                    {
                        "ourPicks": [f"c{number}" for number in range(6)],
                        "opponentPicks": [],
                        "result": "win",
                    }
                ]
            },
            "at most 5",
        ),
        (
            {
                "previousGames": [
                    {"ourPicks": ["c0"], "opponentPicks": ["c0"], "result": "win"},
                ]
            },
            "more than once",
        ),
        ({"previousGames": [_previous(0), _previous(10)]}, "already decided"),
        ({"seriesFormat": "bo1", "previousGames": [_previous(0, "loss")]}, "already decided"),
    ],
)
def test_illegal_draft_states_are_rejected(overrides: dict[str, Any], message: str) -> None:
    with pytest.raises(DraftStateError, match=message):
        build_context(_request(**overrides), CHAMPIONS)
