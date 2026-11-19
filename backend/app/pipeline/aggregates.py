"""Patch-bucketed draft statistics and recent-patch selection.

A game is *decided* when it finished and its winner is one of its two teams. Win-rate
denominators (``decided_games``, ``decided_picks``, side and pair ``games``) count decided
games only, so unfinished games never deflate a win rate.
"""

import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from itertools import combinations

from pydantic import BaseModel, ConfigDict

from app.pipeline.dataset import GameDraft


class ChampionAggregate(BaseModel):
    """Pick, ban, game, and win counts for one champion on a patch.

    ``games`` counts games the champion was picked in; ``decided_games`` is the subset with a
    result and is the win-rate denominator. Presence (picks + bans) is relative to the patch's
    ``game_count``.
    """

    model_config = ConfigDict(frozen=True)

    champion_id: str
    champion_name: str
    games: int
    decided_games: int
    picks: int
    bans: int
    wins: int


class SideAggregate(BaseModel):
    """Blue- and red-side decided game and win counts for a patch."""

    model_config = ConfigDict(frozen=True)

    blue_games: int
    blue_wins: int
    red_games: int
    red_wins: int


class ChampionPairAggregate(BaseModel):
    """Same-team co-pick and win counts for an unordered champion pair in decided games."""

    model_config = ConfigDict(frozen=True)

    champion_a_id: str
    champion_b_id: str
    games: int
    wins: int


class TeamChampionAggregate(BaseModel):
    """A team's pick, win, ban, and against-ban counts for one champion."""

    model_config = ConfigDict(frozen=True)

    team_id: str
    champion_id: str
    picks: int
    decided_picks: int
    wins: int
    bans: int
    banned_against: int


class PatchAggregate(BaseModel):
    """All draft aggregates for a single patch."""

    model_config = ConfigDict(frozen=True)

    patch: str
    game_count: int
    champions: list[ChampionAggregate]
    sides: SideAggregate
    synergies: list[ChampionPairAggregate]
    team_champions: list[TeamChampionAggregate]


@dataclass(slots=True)
class _ChampionCounts:
    champion_name: str
    game_ids: set[str] = field(default_factory=set)
    decided_game_ids: set[str] = field(default_factory=set)
    picks: int = 0
    bans: int = 0
    wins: int = 0


@dataclass(slots=True)
class _PairCounts:
    games: int = 0
    wins: int = 0


@dataclass(slots=True)
class _TeamChampionCounts:
    picks: int = 0
    decided_picks: int = 0
    wins: int = 0
    bans: int = 0
    banned_against: int = 0


UNKNOWN_PATCH = "unknown"
_NUMERIC_PATCH = re.compile(r"\d+(\.\d+)*")


def _patch_label(patch: str) -> str:
    """Return the bucket label for a patch, mapping blank or non-numeric values to ``unknown``."""
    stripped = patch.strip()
    return stripped if _NUMERIC_PATCH.fullmatch(stripped) else UNKNOWN_PATCH


def _patch_sort_key(patch: str) -> tuple[int, ...]:
    """Convert a dotted numeric patch version to a sortable tuple.

    The ``unknown`` bucket (blank or non-numeric patches) sorts older than every numeric patch.
    """
    label = _patch_label(patch)
    if label == UNKNOWN_PATCH:
        return (-1,)
    return tuple(int(part) for part in label.split("."))


def _is_decided(game: GameDraft) -> bool:
    return game.finished and game.winner_team_id in {game.blue_team_id, game.red_team_id}


def compute_aggregates(game_drafts: Sequence[GameDraft]) -> dict[str, PatchAggregate]:
    """Compute champion, side, synergy, and team history counts by patch."""
    games_by_patch: dict[str, list[GameDraft]] = defaultdict(list)
    for game in game_drafts:
        games_by_patch[_patch_label(game.patch)].append(game)

    aggregates: dict[str, PatchAggregate] = {}
    for patch in sorted(games_by_patch, key=_patch_sort_key):
        games = games_by_patch[patch]
        champions: dict[str, _ChampionCounts] = {}
        synergies: dict[tuple[str, str], _PairCounts] = {}
        team_champions: dict[tuple[str, str], _TeamChampionCounts] = {}

        for game in games:
            decided = _is_decided(game)
            picked_by_team: dict[str, set[str]] = defaultdict(set)
            for action in game.actions:
                if action.team_id not in {game.blue_team_id, game.red_team_id}:
                    raise ValueError(
                        f"Draft action has unknown team {action.team_id} in {game.game_id}"
                    )

                champion = champions.setdefault(
                    action.champion_id,
                    _ChampionCounts(champion_name=action.champion_name),
                )
                team_champion = team_champions.setdefault(
                    (action.team_id, action.champion_id),
                    _TeamChampionCounts(),
                )

                if action.action_type == "pick":
                    champion.picks += 1
                    champion.game_ids.add(game.game_id)
                    team_champion.picks += 1
                    picked_by_team[action.team_id].add(action.champion_id)
                    if decided:
                        champion.decided_game_ids.add(game.game_id)
                        team_champion.decided_picks += 1
                        if game.winner_team_id == action.team_id:
                            champion.wins += 1
                            team_champion.wins += 1
                else:
                    champion.bans += 1
                    team_champion.bans += 1
                    opposing_team = (
                        game.red_team_id
                        if action.team_id == game.blue_team_id
                        else game.blue_team_id
                    )
                    against = team_champions.setdefault(
                        (opposing_team, action.champion_id),
                        _TeamChampionCounts(),
                    )
                    against.banned_against += 1

            if not decided:
                continue
            for team_id, picked_champions in picked_by_team.items():
                for champion_a_id, champion_b_id in combinations(sorted(picked_champions), 2):
                    pair = synergies.setdefault(
                        (champion_a_id, champion_b_id),
                        _PairCounts(),
                    )
                    pair.games += 1
                    if game.winner_team_id == team_id:
                        pair.wins += 1

        champion_results = [
            ChampionAggregate(
                champion_id=champion_id,
                champion_name=counts.champion_name,
                games=len(counts.game_ids),
                decided_games=len(counts.decided_game_ids),
                picks=counts.picks,
                bans=counts.bans,
                wins=counts.wins,
            )
            for champion_id, counts in sorted(champions.items())
        ]
        synergy_results = [
            ChampionPairAggregate(
                champion_a_id=pair[0],
                champion_b_id=pair[1],
                games=counts.games,
                wins=counts.wins,
            )
            for pair, counts in sorted(synergies.items())
        ]
        team_results = [
            TeamChampionAggregate(
                team_id=team_champion[0],
                champion_id=team_champion[1],
                picks=counts.picks,
                decided_picks=counts.decided_picks,
                wins=counts.wins,
                bans=counts.bans,
                banned_against=counts.banned_against,
            )
            for team_champion, counts in sorted(team_champions.items())
        ]
        decided_games = [game for game in games if _is_decided(game)]
        aggregates[patch] = PatchAggregate(
            patch=patch,
            game_count=len(games),
            champions=champion_results,
            sides=SideAggregate(
                blue_games=len(decided_games),
                blue_wins=sum(game.winner_team_id == game.blue_team_id for game in decided_games),
                red_games=len(decided_games),
                red_wins=sum(game.winner_team_id == game.red_team_id for game in decided_games),
            ),
            synergies=synergy_results,
            team_champions=team_results,
        )

    return aggregates


def blend_recent(
    patches: Mapping[str, Sequence[GameDraft]],
    min_games: int,
) -> list[GameDraft]:
    """Collect newest patch games until the minimum sample size is reached."""
    if min_games < 1:
        raise ValueError("min_games must be a positive integer")

    selected: list[GameDraft] = []
    for patch in sorted(patches, key=_patch_sort_key, reverse=True):
        selected.extend(patches[patch])
        if len(selected) >= min_games:
            break
    return selected
