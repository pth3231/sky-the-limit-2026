"""Serializable blue-oriented features for historical drafts and live states."""

import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from functools import cached_property
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.engine.draft_order import DRAFT_ORDER
from app.pipeline.dataset import GameDraft
from app.pipeline.ddragon import ChampionTraits, normalize_champion_name
from app.pipeline.drafts import Role

_ROLES: tuple[Role, ...] = ("top", "jungle", "mid", "bot", "support")
_INFO_FIELDS: tuple[str, ...] = ("attack", "defense", "magic", "difficulty")
_META_FIELDS: tuple[str, ...] = ("pick_rate", "ban_rate", "win_rate")
_TEAM_PRIOR_RATE = 0.5
_TEAM_PRIOR_GAMES = 10
_TEAM_RECENT_GAMES_WINDOW = 50
ELO_START = 1500.0
_ELO_K = 48.0
_ELO_SCALE = 400.0
_PLAYER_ELO_K = 24.0
_COMFORT_PRIOR_GAMES = 5.0


def _date_key(game: GameDraft) -> float:
    try:
        value = datetime.fromisoformat(game.started_at.replace("Z", "+00:00"))
        if value.tzinfo is None:
            value = value.replace(tzinfo=UTC)
        return value.timestamp()
    except ValueError:
        return float("-inf")


def team_elo_ratings(
    games: Sequence[GameDraft],
) -> tuple[dict[str, tuple[float, float]], dict[str, float]]:
    """Replay decided games chronologically.

    Returns each game's pre-game (blue, red) Elo keyed by ``game_id`` (no leakage: a game's own
    result is applied only after its ratings are recorded) and the final rating per team.
    """
    ratings: dict[str, float] = {}
    pre_game: dict[str, tuple[float, float]] = {}
    decided = [
        game
        for game in games
        if game.finished and game.winner_team_id in {game.blue_team_id, game.red_team_id}
    ]
    for game in sorted(decided, key=lambda g: (_date_key(g), g.series_id, g.game_number)):
        blue = ratings.get(game.blue_team_id, ELO_START)
        red = ratings.get(game.red_team_id, ELO_START)
        pre_game[game.game_id] = (blue, red)
        expected = 1.0 / (1.0 + 10.0 ** ((red - blue) / _ELO_SCALE))
        delta = _ELO_K * ((1.0 if game.winner_team_id == game.blue_team_id else 0.0) - expected)
        ratings[game.blue_team_id] = blue + delta
        ratings[game.red_team_id] = red - delta
    return pre_game, ratings


@dataclass(frozen=True)
class RosterHistory:
    """Pre-game player strength and player-on-champion record for one game's two rosters."""

    blue_elo: float
    red_elo: float
    blue_comfort: dict[str, tuple[int, int]]  # champion_id -> (games, wins) before this game
    red_comfort: dict[str, tuple[int, int]]


def player_history(games: Sequence[GameDraft]) -> dict[str, RosterHistory]:
    """Replay decided games with player ids chronologically; keyed by ``game_id`` (no leakage).

    Games whose lineups lack five identified players per side are skipped and update nothing.
    """
    elo: dict[str, float] = {}
    record: dict[tuple[str, str], tuple[int, int]] = {}
    history: dict[str, RosterHistory] = {}
    decided = [
        game
        for game in games
        if game.finished and game.winner_team_id in {game.blue_team_id, game.red_team_id}
    ]
    for game in sorted(decided, key=lambda g: (_date_key(g), g.series_id, g.game_number)):
        sides = {
            side: [line for line in game.lineups if line.team_id == team_id and line.player_id]
            for side, team_id in (("blue", game.blue_team_id), ("red", game.red_team_id))
        }
        if len(sides["blue"]) != 5 or len(sides["red"]) != 5:
            continue
        mean = {
            side: sum(elo.get(line.player_id or "", ELO_START) for line in lines) / 5
            for side, lines in sides.items()
        }
        history[game.game_id] = RosterHistory(
            blue_elo=mean["blue"],
            red_elo=mean["red"],
            blue_comfort={
                line.champion_id: record.get((line.player_id or "", line.champion_id), (0, 0))
                for line in sides["blue"]
            },
            red_comfort={
                line.champion_id: record.get((line.player_id or "", line.champion_id), (0, 0))
                for line in sides["red"]
            },
        )
        blue_won = game.winner_team_id == game.blue_team_id
        expected = 1.0 / (1.0 + 10.0 ** ((mean["red"] - mean["blue"]) / _ELO_SCALE))
        delta = _PLAYER_ELO_K * ((1.0 if blue_won else 0.0) - expected)
        for side, lines in sides.items():
            won = blue_won if side == "blue" else not blue_won
            for line in lines:
                key = (line.player_id or "", line.champion_id)
                elo[key[0]] = elo.get(key[0], ELO_START) + (delta if side == "blue" else -delta)
                games_played, wins = record.get(key, (0, 0))
                record[key] = (games_played + 1, wins + int(won))
    return history


class DraftState(BaseModel):
    """Blue-oriented partial or complete draft state used by the model."""

    model_config = ConfigDict(frozen=True)

    patch: str = "unknown"
    actions: list[str] = Field(default_factory=list)
    blue_team_id: str | None = None
    red_team_id: str | None = None
    # Pre-game Elo; when None, encode falls back to the fitted final rating for the team id.
    blue_elo: float | None = None
    red_elo: float | None = None
    # Mean pre-game player Elo per roster, and each pick's (games, wins) for its player.
    blue_player_elo: float | None = None
    red_player_elo: float | None = None
    blue_comfort: dict[str, tuple[int, int]] = Field(default_factory=dict)
    red_comfort: dict[str, tuple[int, int]] = Field(default_factory=dict)


@dataclass(frozen=True)
class EncodedState:
    """Champion-index slot arrays plus continuous aggregate features for one draft state."""

    blue_picks: list[int]
    blue_bans: list[int]
    red_picks: list[int]
    red_bans: list[int]
    continuous: list[float] = field(default_factory=list)


class ChampionMeta(BaseModel):
    """Training-set pick, ban, and win rates for one champion."""

    model_config = ConfigDict(frozen=True)

    pick_rate: float
    ban_rate: float
    win_rate: float


class RoleProfile(BaseModel):
    """Training-set win and player-performance profile for a champion role."""

    model_config = ConfigDict(frozen=True)

    games: int
    win_rate: float
    strength: float
    damage_share: float
    damage_taken: float
    vision_score: float
    gold_share: float
    kill_participation: float


def _role_strength(
    damage_share: float,
    gold_share: float,
    kill_participation: float,
    vision_score: float,
) -> float:
    return (
        sum(
            min(max(value, 0.0), 1.0)
            for value in (damage_share, gold_share, kill_participation, vision_score / 100.0)
        )
        / 4.0
    )


def _champion_meta(
    counts: dict[str, list[int]],
    game_count: int,
) -> dict[str, ChampionMeta]:
    result: dict[str, ChampionMeta] = {}
    for champion_id, values in counts.items():
        picks, bans, decided_picks, wins = values
        result[champion_id] = ChampionMeta(
            pick_rate=picks / game_count if game_count else 0.0,
            ban_rate=bans / game_count if game_count else 0.0,
            win_rate=wins / decided_picks if decided_picks else 0.5,
        )
    return result


def _coerce_traits(value: Any) -> ChampionTraits | None:
    if isinstance(value, ChampionTraits):
        return value
    if isinstance(value, Mapping):
        try:
            return ChampionTraits.model_validate(value)
        except ValueError:
            return None
    return None


class FeatureSpace(BaseModel):
    """Fitted vocabularies and historical signals used to encode draft states."""

    model_config = ConfigDict(frozen=True)

    champion_ids: list[str]
    patches: list[str]
    tag_names: list[str]
    champion_roles: dict[str, Role]
    champion_traits: dict[str, ChampionTraits]
    patch_champion_stats: dict[str, dict[str, ChampionMeta]]
    global_champion_stats: dict[str, ChampionMeta]
    role_profiles: dict[str, dict[Role, RoleProfile]]
    matchup_win_rates: dict[str, float] = Field(default_factory=dict)
    team_win_rates: dict[str, float] = Field(default_factory=dict)
    team_elo: dict[str, float] = Field(default_factory=dict)

    @classmethod
    def fit(
        cls,
        train_games: Sequence[GameDraft],
        ddragon: Mapping[str, Any] | None = None,
        *,
        vocabulary_games: Sequence[GameDraft] | None = None,
    ) -> "FeatureSpace":
        """Fit feature vocabularies and rates from training games only."""
        champions: set[str] = set()
        champion_names: dict[str, str] = {}
        vocabulary_patches: set[str] = set()
        patch_counts: dict[str, dict[str, list[int]]] = {}
        patch_game_counts: Counter[str] = Counter()
        global_counts: dict[str, list[int]] = {}
        global_game_count = 0
        role_counts: dict[str, Counter[Role]] = {}
        role_values: dict[str, dict[Role, list[float]]] = {}
        matchup_counts: dict[tuple[Role, str, str], list[int]] = {}
        team_game_log: dict[str, list[tuple[float, bool]]] = {}

        normalized_traits: dict[str, ChampionTraits] = {}
        for name, value in (ddragon or {}).items():
            traits = _coerce_traits(value)
            if traits is not None:
                normalized_traits[normalize_champion_name(name)] = traits

        vocabulary = train_games if vocabulary_games is None else vocabulary_games
        for game in vocabulary:
            vocabulary_patches.add(game.patch.strip() or "unknown")
            for action in game.actions:
                champions.add(action.champion_id)
                champion_names.setdefault(action.champion_id, action.champion_name)
            for player in game.lineups:
                champions.add(player.champion_id)

        for game in train_games:
            patch = game.patch.strip() or "unknown"
            patch_game_counts[patch] += 1
            global_game_count += 1
            patch_stats = patch_counts.setdefault(patch, {})
            decided = game.finished and game.winner_team_id in {
                game.blue_team_id,
                game.red_team_id,
            }
            if decided:
                for team_id in (game.blue_team_id, game.red_team_id):
                    team_game_log.setdefault(team_id, []).append(
                        (_date_key(game), team_id == game.winner_team_id)
                    )
            for action in game.actions:
                champions.add(action.champion_id)
                champion_names.setdefault(action.champion_id, action.champion_name)
                patch_values = patch_stats.setdefault(action.champion_id, [0, 0, 0, 0])
                global_values = global_counts.setdefault(action.champion_id, [0, 0, 0, 0])
                index = 0 if action.action_type == "pick" else 1
                patch_values[index] += 1
                global_values[index] += 1
                if action.action_type == "pick" and decided:
                    patch_values[2] += 1
                    global_values[2] += 1
                    if action.team_id == game.winner_team_id:
                        patch_values[3] += 1
                        global_values[3] += 1

            for player in game.lineups:
                champions.add(player.champion_id)
                if player.role is None:
                    continue
                usage = role_counts.setdefault(player.champion_id, Counter())
                usage[player.role] += 1
                values = role_values.setdefault(player.champion_id, {}).setdefault(
                    player.role,
                    [0.0] * 9,
                )
                values[0] += 1
                if decided:
                    values[1] += 1
                    if player.team_id == game.winner_team_id:
                        values[2] += 1
                values[3] += player.damage_share
                values[4] += player.damage_taken
                values[5] += player.vision_score
                values[6] += player.gold_share
                values[7] += player.kill_participation
                values[8] += _role_strength(
                    player.damage_share,
                    player.gold_share,
                    player.kill_participation,
                    player.vision_score,
                )

            lineup_by_role: dict[Role, dict[str, str]] = {}
            for player in game.lineups:
                if player.role is None:
                    continue
                lineup_by_role.setdefault(player.role, {})[player.team_id] = player.champion_id
            if decided:
                for role, by_team in lineup_by_role.items():
                    blue_champion = by_team.get(game.blue_team_id)
                    red_champion = by_team.get(game.red_team_id)
                    if blue_champion is None or red_champion is None:
                        continue
                    blue_won = game.winner_team_id == game.blue_team_id
                    forward = matchup_counts.setdefault((role, blue_champion, red_champion), [0, 0])
                    forward[0] += 1
                    forward[1] += blue_won
                    backward = matchup_counts.setdefault(
                        (role, red_champion, blue_champion), [0, 0]
                    )
                    backward[0] += 1
                    backward[1] += not blue_won

        champion_traits: dict[str, ChampionTraits] = {}
        for champion_id, name in champion_names.items():
            traits = normalized_traits.get(normalize_champion_name(name))
            if traits is not None:
                champion_traits[champion_id] = traits

        champion_roles: dict[str, Role] = {}
        role_profiles: dict[str, dict[Role, RoleProfile]] = {}
        for champion_id, counts in role_counts.items():
            champion_roles[champion_id] = max(
                _ROLES,
                key=lambda role: (counts[role], -_ROLES.index(role)),
            )
            role_profiles[champion_id] = {}
            for role, values in role_values[champion_id].items():
                games = int(values[0])
                decided_games = values[1]
                role_profiles[champion_id][role] = RoleProfile(
                    games=games,
                    win_rate=values[2] / decided_games if decided_games else 0.5,
                    strength=values[8] / games if games else 0.0,
                    damage_share=values[3] / games if games else 0.0,
                    damage_taken=values[4] / games if games else 0.0,
                    vision_score=values[5] / games if games else 0.0,
                    gold_share=values[6] / games if games else 0.0,
                    kill_participation=values[7] / games if games else 0.0,
                )

        matchup_win_rates = {
            f"{role}|{champion_id}|{opponent_id}": wins / games
            for (role, champion_id, opponent_id), (games, wins) in matchup_counts.items()
        }
        team_win_rates: dict[str, float] = {}
        for team_id, entries in team_game_log.items():
            recent = sorted(entries, key=lambda entry: entry[0], reverse=True)
            recent = recent[:_TEAM_RECENT_GAMES_WINDOW]
            games = len(recent)
            wins = sum(1 for _, won in recent if won)
            team_win_rates[team_id] = (wins + _TEAM_PRIOR_RATE * _TEAM_PRIOR_GAMES) / (
                games + _TEAM_PRIOR_GAMES
            )

        return cls(
            champion_ids=sorted(champions),
            patches=sorted(patch for patch in vocabulary_patches if patch != "unknown"),
            tag_names=sorted({tag for traits in champion_traits.values() for tag in traits.tags}),
            champion_roles=champion_roles,
            champion_traits=champion_traits,
            patch_champion_stats={
                patch: _champion_meta(counts, patch_game_counts[patch])
                for patch, counts in patch_counts.items()
            },
            global_champion_stats=_champion_meta(global_counts, global_game_count),
            role_profiles=role_profiles,
            matchup_win_rates=matchup_win_rates,
            team_win_rates=team_win_rates,
            team_elo=team_elo_ratings(train_games)[1],
        )

    @property
    def num_champions(self) -> int:
        """Return the fitted champion vocabulary size (the embedding padding index)."""
        return len(self.champion_ids)

    @property
    def continuous_size(self) -> int:
        """Return the stable width of the continuous feature block for this feature space."""
        return len(self.encode(DraftState()).continuous)

    @cached_property
    def _champion_index(self) -> dict[str, int]:
        """Champion id -> embedding index, built once per instance instead of per encode() call."""
        return {champion_id: index for index, champion_id in enumerate(self.champion_ids)}

    def _meta_for(self, patch: str, champion_id: str) -> ChampionMeta | None:
        return self.patch_champion_stats.get(patch, {}).get(
            champion_id,
            self.global_champion_stats.get(champion_id),
        )

    def encode(self, state: DraftState) -> EncodedState:
        """Encode a draft state as champion-index slot arrays plus continuous features."""
        champion_index = self._champion_index
        sentinel = self.num_champions
        slots: dict[str, list[int]] = {
            "blue_picks": [sentinel] * 5,
            "blue_bans": [sentinel] * 5,
            "red_picks": [sentinel] * 5,
            "red_bans": [sentinel] * 5,
        }
        slot_cursors = {key: 0 for key in slots}
        picks: dict[str, list[str]] = {"blue": [], "red": []}
        bans: dict[str, list[str]] = {"blue": [], "red": []}

        for index, champion_id in enumerate(state.actions[: len(DRAFT_ORDER)]):
            slot = DRAFT_ORDER[index]
            group_key = f"{slot.side}_{'picks' if slot.action == 'pick' else 'bans'}"
            cursor = slot_cursors[group_key]
            if cursor < len(slots[group_key]):
                champion_position = champion_index.get(champion_id)
                if champion_position is not None:
                    slots[group_key][cursor] = champion_position
                slot_cursors[group_key] = cursor + 1
            selected = picks if slot.action == "pick" else bans
            selected[slot.side].append(champion_id)

        action_count = [0.0] * (len(DRAFT_ORDER) + 1)
        action_count[min(len(state.actions), len(DRAFT_ORDER))] = 1.0
        patch_features = [0.0] * (len(self.patches) + 1)
        patch_index = (
            self.patches.index(state.patch) if state.patch in self.patches else len(self.patches)
        )
        patch_features[patch_index] = 1.0
        continuous = [*action_count, *patch_features]
        patch_key = state.patch.strip() or "unknown"

        for side in ("blue", "red"):
            current_picks = picks[side]
            role_counts = Counter(
                self.champion_roles[champion_id]
                for champion_id in current_picks
                if champion_id in self.champion_roles
            )
            continuous.extend(role_counts[role] / 5.0 for role in _ROLES)

            profiles: list[RoleProfile] = []
            for champion_id in current_picks:
                role = self.champion_roles.get(champion_id)
                if role is None:
                    continue
                profile = self.role_profiles.get(champion_id, {}).get(role)
                if profile is not None:
                    profiles.append(profile)
            profile_values = (
                [profile.win_rate for profile in profiles],
                [profile.strength for profile in profiles],
                [profile.damage_share for profile in profiles],
                [min(profile.damage_taken / 100_000.0, 1.0) for profile in profiles],
                [min(profile.vision_score / 100.0, 1.0) for profile in profiles],
                [profile.gold_share for profile in profiles],
                [profile.kill_participation for profile in profiles],
            )
            continuous.extend(
                sum(values) / len(values) if values else 0.0 for values in profile_values
            )

            side_traits = [self.champion_traits.get(champion_id) for champion_id in current_picks]
            side_traits = [traits for traits in side_traits if traits is not None]
            continuous.extend(
                sum(tag in traits.tags for traits in side_traits) / 5.0 for tag in self.tag_names
            )
            continuous.extend(
                sum(traits.info.get(field, 0.0) for traits in side_traits)
                / (10.0 * len(side_traits))
                if side_traits
                else 0.0
                for field in _INFO_FIELDS
            )

            for selected in (current_picks, bans[side]):
                for meta_field in _META_FIELDS:
                    values = [
                        getattr(meta, meta_field)
                        for champion_id in selected
                        if (meta := self._meta_for(patch_key, champion_id)) is not None
                    ]
                    continuous.append(sum(values) / len(values) if values else 0.0)

            opponent_side = "red" if side == "blue" else "blue"
            opponent_picks = picks[opponent_side]
            matchup_rates: list[float] = []
            for champion_id in current_picks:
                champion_role = self.champion_roles.get(champion_id)
                if champion_role is None:
                    continue
                opponent_id = next(
                    (
                        cid
                        for cid in opponent_picks
                        if self.champion_roles.get(cid) == champion_role
                    ),
                    None,
                )
                if opponent_id is None:
                    continue
                rate = self.matchup_win_rates.get(f"{champion_role}|{champion_id}|{opponent_id}")
                if rate is not None:
                    matchup_rates.append(rate)
            continuous.append(sum(matchup_rates) / len(matchup_rates) if matchup_rates else 0.0)

            team_id = state.blue_team_id if side == "blue" else state.red_team_id
            continuous.append(
                self.team_win_rates.get(team_id, _TEAM_PRIOR_RATE) if team_id else _TEAM_PRIOR_RATE
            )

        for elo, team_id in (
            (state.blue_elo, state.blue_team_id),
            (state.red_elo, state.red_team_id),
        ):
            if elo is None:
                elo = self.team_elo.get(team_id, ELO_START) if team_id else ELO_START
            continuous.append((elo - ELO_START) / _ELO_SCALE)

        for side, player_elo, comfort in (
            ("blue", state.blue_player_elo, state.blue_comfort),
            ("red", state.red_player_elo, state.red_comfort),
        ):
            continuous.append(((player_elo or ELO_START) - ELO_START) / _ELO_SCALE)
            # Only picks already revealed count, so prefix states never see future picks.
            stats = [comfort[champion_id] for champion_id in picks[side] if champion_id in comfort]
            prior = _COMFORT_PRIOR_GAMES
            continuous.append(
                sum((w + 0.5 * prior) / (g + prior) - 0.5 for g, w in stats) / len(stats)
                if stats
                else 0.0
            )
            continuous.append(sum(math.log1p(g) for g, _ in stats) / len(stats) if stats else 0.0)

        return EncodedState(
            blue_picks=slots["blue_picks"],
            blue_bans=slots["blue_bans"],
            red_picks=slots["red_picks"],
            red_bans=slots["red_bans"],
            continuous=continuous,
        )
