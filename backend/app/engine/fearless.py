"""Fearless series rules and validation of the submitted draft state."""

from collections.abc import Collection, Iterable, Sequence

from pydantic import BaseModel, ConfigDict

from app.engine.draft_order import DRAFT_ORDER, DraftSlot
from app.engine.errors import DraftStateError
from app.engine.models import DraftRequest, PreviousGame, SeriesFormat, Side

_MAX_PICKS_PER_SIDE = 5
_GAMES_IN_FORMAT: dict[SeriesFormat, int] = {"bo1": 1, "bo3": 3, "bo5": 5}


class DraftContext(BaseModel):
    """A validated draft state split by team and action."""

    model_config = ConfigDict(frozen=True)

    game_number: int
    series_format: SeriesFormat
    our_side: Side
    series_score: tuple[int, int]
    our_picks: tuple[str, ...]
    opponent_picks: tuple[str, ...]
    our_bans: tuple[str, ...]
    opponent_bans: tuple[str, ...]
    next_slot: DraftSlot | None
    fearless_excluded: frozenset[str]
    unavailable: frozenset[str]


def wins_needed(series_format: SeriesFormat) -> int:
    """Return the number of game wins that decides a series of the given format."""
    return _GAMES_IN_FORMAT[series_format] // 2 + 1


def fearless_excluded(previous_games: Iterable[PreviousGame]) -> set[str]:
    """Return every champion picked by either team in an earlier game; bans do not carry over."""
    excluded: set[str] = set()
    for game in previous_games:
        excluded.update(game.our_picks)
        excluded.update(game.opponent_picks)
    return excluded


def _require_known(champion_ids: Iterable[str], known: Collection[str]) -> None:
    for champion_id in champion_ids:
        if champion_id not in known:
            raise DraftStateError(f"Unknown champion: {champion_id}")


def _require_unique(champion_ids: Sequence[str], where: str) -> None:
    seen: set[str] = set()
    for champion_id in champion_ids:
        if champion_id in seen:
            raise DraftStateError(f"Champion {champion_id} appears more than once in {where}")
        seen.add(champion_id)


def _validate_previous_games(request: DraftRequest, known: Collection[str]) -> tuple[int, int]:
    needed = wins_needed(request.series_format)
    wins = losses = 0
    earlier_picks: set[str] = set()
    for number, game in enumerate(request.previous_games, start=1):
        if wins >= needed or losses >= needed:
            raise DraftStateError(f"The series was already decided before game {number}")
        if (
            len(game.our_picks) > _MAX_PICKS_PER_SIDE
            or len(game.opponent_picks) > _MAX_PICKS_PER_SIDE
        ):
            raise DraftStateError(f"Game {number} can have at most 5 picks per team")
        if (
            len(game.our_picks) != _MAX_PICKS_PER_SIDE
            or len(game.opponent_picks) != _MAX_PICKS_PER_SIDE
        ):
            raise DraftStateError(f"Completed game {number} must have exactly 5 picks per team")
        picks = [*game.our_picks, *game.opponent_picks]
        _require_known(picks, known)
        _require_unique(picks, f"game {number}")
        repeated = earlier_picks.intersection(picks)
        if repeated:
            raise DraftStateError(
                f"Game {number} repeats Fearless-excluded champions: {', '.join(sorted(repeated))}"
            )
        earlier_picks.update(picks)
        if game.result == "win":
            wins += 1
        else:
            losses += 1

    if wins >= needed or losses >= needed:
        raise DraftStateError("The series is already decided; no further game can be drafted")
    return wins, losses


def build_context(request: DraftRequest, champion_ids: Collection[str]) -> DraftContext:
    """Validate the request against draft and Fearless rules and split it by team and action."""
    series_score = _validate_previous_games(request, champion_ids)
    excluded = frozenset(fearless_excluded(request.previous_games))

    if len(request.actions) > len(DRAFT_ORDER):
        raise DraftStateError(f"A draft has at most {len(DRAFT_ORDER)} actions")
    _require_known(request.actions, champion_ids)
    _require_unique(request.actions, "the current draft")
    blocked = excluded.intersection(request.actions)
    if blocked:
        raise DraftStateError(
            f"Fearless-excluded champions cannot be drafted: {', '.join(sorted(blocked))}"
        )

    picks: dict[Side, list[str]] = {"blue": [], "red": []}
    bans: dict[Side, list[str]] = {"blue": [], "red": []}
    for slot, champion_id in zip(DRAFT_ORDER, request.actions, strict=False):
        (picks if slot.action == "pick" else bans)[slot.side].append(champion_id)

    our_side = request.our_side
    their_side: Side = "red" if our_side == "blue" else "blue"
    next_index = len(request.actions)
    return DraftContext(
        game_number=len(request.previous_games) + 1,
        series_format=request.series_format,
        our_side=our_side,
        series_score=series_score,
        our_picks=tuple(picks[our_side]),
        opponent_picks=tuple(picks[their_side]),
        our_bans=tuple(bans[our_side]),
        opponent_bans=tuple(bans[their_side]),
        next_slot=DRAFT_ORDER[next_index] if next_index < len(DRAFT_ORDER) else None,
        fearless_excluded=excluded,
        unavailable=excluded.union(request.actions),
    )
