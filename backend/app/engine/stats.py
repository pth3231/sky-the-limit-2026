"""Blend recent patch aggregates into summed lookups for scoring."""

from collections.abc import Iterable, Mapping

from pydantic import BaseModel, ConfigDict

from app.engine.errors import UnknownPatchError
from app.engine.models import Side
from app.pipeline.aggregates import (
    ChampionAggregate,
    ChampionMatchupAggregate,
    ChampionPairAggregate,
    PatchAggregate,
    TeamAggregate,
    TeamChampionAggregate,
    patch_sort_key,
)

DEFAULT_MIN_GAMES = 60
MIN_SUBJECT_GAMES = 10


class BlendedStats(BaseModel):
    """Counts summed over the patches selected for a recommendation."""

    model_config = ConfigDict(frozen=True)

    patches: tuple[str, ...]
    game_count: int
    blue_wins: int
    red_wins: int
    decided_games: int
    champions: Mapping[str, ChampionAggregate]
    champion_patch_map: Mapping[str, tuple[str, ...]]
    pairs: Mapping[tuple[str, str], ChampionPairAggregate]
    matchups: Mapping[tuple[str, str, str], ChampionMatchupAggregate]
    team_champions: Mapping[tuple[str, str], TeamChampionAggregate]
    teams: Mapping[str, TeamAggregate]
    team_patch_map: Mapping[str, tuple[str, ...]]

    def champion(self, champion_id: str) -> ChampionAggregate | None:
        """Return summed counts for a champion, or ``None`` without data."""
        return self.champions.get(champion_id)

    def pair(self, champion_a_id: str, champion_b_id: str) -> ChampionPairAggregate | None:
        """Return same-team counts for an unordered champion pair, or ``None`` without data."""
        first, second = sorted((champion_a_id, champion_b_id))
        return self.pairs.get((first, second))

    def matchup(
        self, role: str, champion_id: str, opponent_champion_id: str
    ) -> ChampionMatchupAggregate | None:
        """Return a champion's per-role record against a specific opposing champion."""
        return self.matchups.get((role, champion_id, opponent_champion_id))

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

    def team_patches(self, team_id: str) -> tuple[str, ...]:
        """Return the recent-first patches used for this team's history."""
        return self.team_patch_map.get(team_id, ())

    def all_patches(self) -> tuple[str, ...]:
        """Return every patch contributing to global or subject-specific recommendation data."""
        patches = set(self.patches)
        patches.update(label for labels in self.champion_patch_map.values() for label in labels)
        patches.update(label for labels in self.team_patch_map.values() for label in labels)
        return tuple(sorted(patches, key=patch_sort_key, reverse=True))

    def side_record(self, side: Side) -> tuple[int, int]:
        """Return ``(wins, decided games)`` for a side."""
        wins = self.blue_wins if side == "blue" else self.red_wins
        return wins, self.decided_games


def _eligible_patches(
    aggregates: Mapping[str, PatchAggregate],
    patch: str | None,
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

    return [aggregates[label] for label in ordered[start:]]


def _select_patches(
    aggregates: Mapping[str, PatchAggregate],
    patch: str | None,
    min_games: int,
) -> list[PatchAggregate]:
    eligible = _eligible_patches(aggregates, patch)
    selected: list[PatchAggregate] = []
    total = 0
    for aggregate in eligible:
        selected.append(aggregate)
        total += aggregate.game_count
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


def _sum_matchups(
    patches: Iterable[PatchAggregate],
) -> dict[tuple[str, str, str], ChampionMatchupAggregate]:
    summed: dict[tuple[str, str, str], ChampionMatchupAggregate] = {}
    for patch in patches:
        for item in patch.matchups:
            key = (item.role, item.champion_id, item.opponent_champion_id)
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


def _sum_subject_champions(
    patches: list[PatchAggregate],
) -> tuple[dict[str, ChampionAggregate], dict[str, tuple[str, ...]]]:
    champion_ids = {item.champion_id for patch in patches for item in patch.champions}
    summed: dict[str, ChampionAggregate] = {}
    champion_patch_map: dict[str, tuple[str, ...]] = {}
    for champion_id in champion_ids:
        subject_patches: list[PatchAggregate] = []
        games = 0
        for patch in patches:
            record = next(
                (item for item in patch.champions if item.champion_id == champion_id), None
            )
            if record is None:
                continue
            subject_patches.append(patch)
            games += record.decided_games
            if games >= MIN_SUBJECT_GAMES:
                break
        summed[champion_id] = _sum_champions(subject_patches)[champion_id]
        champion_patch_map[champion_id] = tuple(patch.patch for patch in subject_patches)
    return summed, champion_patch_map


def _sum_subject_teams(
    patches: list[PatchAggregate],
) -> tuple[
    dict[str, TeamAggregate],
    dict[tuple[str, str], TeamChampionAggregate],
    dict[str, tuple[str, ...]],
]:
    team_ids = {item.team_id for patch in patches for item in patch.teams}
    teams: dict[str, TeamAggregate] = {}
    team_champions: dict[tuple[str, str], TeamChampionAggregate] = {}
    team_patch_map: dict[str, tuple[str, ...]] = {}
    for team_id in team_ids:
        subject_patches: list[PatchAggregate] = []
        games = 0
        for patch in patches:
            record = next((item for item in patch.teams if item.team_id == team_id), None)
            if record is None:
                continue
            subject_patches.append(patch)
            games += record.games
            if games >= MIN_SUBJECT_GAMES:
                break
        team_patch_map[team_id] = tuple(patch.patch for patch in subject_patches)
        teams.update({team_id: _sum_teams(subject_patches)[team_id]} if subject_patches else {})
        records = _sum_team_champions(subject_patches)
        team_champions.update({key: value for key, value in records.items() if key[0] == team_id})
    return teams, team_champions, team_patch_map


def blend_patches(
    aggregates: Mapping[str, PatchAggregate],
    patch: str | None = None,
    min_games: int = DEFAULT_MIN_GAMES,
) -> BlendedStats:
    """Sum the requested patch (default: newest) and older patches until ``min_games`` is met.

    Patches newer than the requested one are never used. Raises ``UnknownPatchError`` when the
    requested patch has no aggregates.
    """
    eligible = _eligible_patches(aggregates, patch)
    selected = _select_patches(aggregates, patch, min_games)
    champions, champion_patch_map = _sum_subject_champions(eligible)
    teams, team_champions, team_patch_map = _sum_subject_teams(eligible)
    return BlendedStats(
        patches=tuple(item.patch for item in selected),
        game_count=sum(item.game_count for item in selected),
        blue_wins=sum(item.sides.blue_wins for item in selected),
        red_wins=sum(item.sides.red_wins for item in selected),
        decided_games=sum(item.sides.blue_games for item in selected),
        champions=champions,
        champion_patch_map=champion_patch_map,
        pairs=_sum_pairs(selected),
        matchups=_sum_matchups(selected),
        team_champions=team_champions,
        teams=teams,
        team_patch_map=team_patch_map,
    )
