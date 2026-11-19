"""Blend recent patch aggregates into summed lookups for scoring."""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from app.engine.errors import UnknownPatchError
from app.engine.models import Side
from app.pipeline.aggregates import (
    ChampionAggregate,
    ChampionPairAggregate,
    PatchAggregate,
    TeamAggregate,
    TeamChampionAggregate,
    patch_sort_key,
)

DEFAULT_MIN_GAMES = 60


@dataclass(frozen=True, slots=True)
class BlendedStats:
    """Counts summed over the patches selected for a recommendation."""

    patches: tuple[str, ...]
    game_count: int
    blue_wins: int
    red_wins: int
    decided_games: int
    champions: Mapping[str, ChampionAggregate]
    pairs: Mapping[tuple[str, str], ChampionPairAggregate]
    team_champions: Mapping[tuple[str, str], TeamChampionAggregate]
    teams: Mapping[str, TeamAggregate]

    def champion(self, champion_id: str) -> ChampionAggregate | None:
        """Return summed counts for a champion, or ``None`` without data."""
        return self.champions.get(champion_id)

    def pair(self, champion_a_id: str, champion_b_id: str) -> ChampionPairAggregate | None:
        """Return same-team counts for an unordered champion pair, or ``None`` without data."""
        first, second = sorted((champion_a_id, champion_b_id))
        return self.pairs.get((first, second))

    def team_champion(self, team_id: str, champion_id: str) -> TeamChampionAggregate | None:
        """Return a team's history with a champion, or ``None`` without data."""
        return self.team_champions.get((team_id, champion_id))

    def team_history(self, team_id: str) -> list[TeamChampionAggregate]:
        """Return every champion record of a team."""
        return [record for (team, _), record in self.team_champions.items() if team == team_id]

    def team_games(self, team_id: str) -> int:
        """Return how many games the team played on the blended patches."""
        record = self.teams.get(team_id)
        return record.games if record else 0

    def side_record(self, side: Side) -> tuple[int, int]:
        """Return ``(wins, decided games)`` for a side."""
        wins = self.blue_wins if side == "blue" else self.red_wins
        return wins, self.decided_games


def _select_patches(
    aggregates: Mapping[str, PatchAggregate],
    patch: str | None,
    min_games: int,
) -> list[PatchAggregate]:
    if not aggregates:
        raise UnknownPatchError("No patch aggregates are available; build the derived data first")
    ordered = sorted(aggregates, key=patch_sort_key, reverse=True)
    if patch is None:
        start = 0
    elif patch in aggregates:
        start = ordered.index(patch)
    else:
        raise UnknownPatchError(f"No data for patch {patch}")

    selected: list[PatchAggregate] = []
    total = 0
    for label in ordered[start:]:
        selected.append(aggregates[label])
        total += aggregates[label].game_count
        if total >= min_games:
            break
    return selected


def _sum_champions(patches: Iterable[PatchAggregate]) -> dict[str, ChampionAggregate]:
    summed: dict[str, ChampionAggregate] = {}
    for patch in patches:
        for item in patch.champions:
            previous = summed.get(item.champion_id)
            if previous is None:
                summed[item.champion_id] = item
                continue
            summed[item.champion_id] = previous.model_copy(
                update={
                    "games": previous.games + item.games,
                    "decided_games": previous.decided_games + item.decided_games,
                    "picks": previous.picks + item.picks,
                    "bans": previous.bans + item.bans,
                    "wins": previous.wins + item.wins,
                }
            )
    return summed


def _sum_pairs(
    patches: Iterable[PatchAggregate],
) -> dict[tuple[str, str], ChampionPairAggregate]:
    summed: dict[tuple[str, str], ChampionPairAggregate] = {}
    for patch in patches:
        for item in patch.synergies:
            key = (item.champion_a_id, item.champion_b_id)
            previous = summed.get(key)
            summed[key] = (
                item
                if previous is None
                else previous.model_copy(
                    update={"games": previous.games + item.games, "wins": previous.wins + item.wins}
                )
            )
    return summed


def _sum_team_champions(
    patches: Iterable[PatchAggregate],
) -> dict[tuple[str, str], TeamChampionAggregate]:
    summed: dict[tuple[str, str], TeamChampionAggregate] = {}
    for patch in patches:
        for item in patch.team_champions:
            key = (item.team_id, item.champion_id)
            previous = summed.get(key)
            if previous is None:
                summed[key] = item
                continue
            summed[key] = previous.model_copy(
                update={
                    "picks": previous.picks + item.picks,
                    "decided_picks": previous.decided_picks + item.decided_picks,
                    "wins": previous.wins + item.wins,
                    "bans": previous.bans + item.bans,
                    "banned_against": previous.banned_against + item.banned_against,
                }
            )
    return summed


def _sum_teams(patches: Iterable[PatchAggregate]) -> dict[str, TeamAggregate]:
    summed: dict[str, TeamAggregate] = {}
    for patch in patches:
        for item in patch.teams:
            previous = summed.get(item.team_id)
            summed[item.team_id] = (
                item
                if previous is None
                else previous.model_copy(
                    update={
                        "games": previous.games + item.games,
                        "decided_games": previous.decided_games + item.decided_games,
                        "wins": previous.wins + item.wins,
                    }
                )
            )
    return summed


def blend_patches(
    aggregates: Mapping[str, PatchAggregate],
    patch: str | None = None,
    min_games: int = DEFAULT_MIN_GAMES,
) -> BlendedStats:
    """Sum the requested patch (default: newest) and older patches until ``min_games`` is met.

    Patches newer than the requested one are never used. Raises ``UnknownPatchError`` when the
    requested patch has no aggregates.
    """
    selected = _select_patches(aggregates, patch, min_games)
    return BlendedStats(
        patches=tuple(item.patch for item in selected),
        game_count=sum(item.game_count for item in selected),
        blue_wins=sum(item.sides.blue_wins for item in selected),
        red_wins=sum(item.sides.red_wins for item in selected),
        decided_games=sum(item.sides.blue_games for item in selected),
        champions=_sum_champions(selected),
        pairs=_sum_pairs(selected),
        team_champions=_sum_team_champions(selected),
        teams=_sum_teams(selected),
    )
