"""Serializable blue-oriented features for historical drafts and live states."""

from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.engine.draft_order import DRAFT_ORDER
from app.pipeline.dataset import GameDraft
from app.pipeline.ddragon import ChampionTraits, normalize_champion_name
from app.pipeline.drafts import Role

_ROLES: tuple[Role, ...] = ("top", "jungle", "mid", "bot", "support")
_INFO_FIELDS: tuple[str, ...] = ("attack", "defense", "magic", "difficulty")
_META_FIELDS: tuple[str, ...] = ("pick_rate", "ban_rate", "win_rate")


class DraftState(BaseModel):
    """Blue-oriented partial or complete draft state used by the model."""

    model_config = ConfigDict(frozen=True)

    patch: str = "unknown"
    actions: list[str] = Field(default_factory=list)


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
        )

    @property
    def input_size(self) -> int:
        """Return the stable encoded vector width for this fitted feature space."""
        return len(self.encode(DraftState()))

    def _meta_for(self, patch: str, champion_id: str) -> ChampionMeta | None:
        return self.patch_champion_stats.get(patch, {}).get(
            champion_id,
            self.global_champion_stats.get(champion_id),
        )

    def encode(self, state: DraftState) -> list[float]:
        """Encode draft actions with blue/red, pick/ban, role, trait, and meta features."""
        champion_index = {champion_id: index for index, champion_id in enumerate(self.champion_ids)}
        champion_count = len(self.champion_ids)
        presence = [0.0] * (4 * champion_count)
        picks: dict[str, list[str]] = {"blue": [], "red": []}
        bans: dict[str, list[str]] = {"blue": [], "red": []}

        for index, champion_id in enumerate(state.actions[: len(DRAFT_ORDER)]):
            slot = DRAFT_ORDER[index]
            side_index = 0 if slot.side == "blue" else 1
            action_index = 0 if slot.action == "pick" else 1
            channel = side_index * 2 + action_index
            champion_position = champion_index.get(champion_id)
            if champion_position is not None:
                presence[channel * champion_count + champion_position] = 1.0
            selected = picks if slot.action == "pick" else bans
            selected[slot.side].append(champion_id)

        action_count = [0.0] * (len(DRAFT_ORDER) + 1)
        action_count[min(len(state.actions), len(DRAFT_ORDER))] = 1.0
        patch_features = [0.0] * (len(self.patches) + 1)
        patch_index = (
            self.patches.index(state.patch) if state.patch in self.patches else len(self.patches)
        )
        patch_features[patch_index] = 1.0
        features = [*presence, *action_count, *patch_features]
        patch_key = state.patch.strip() or "unknown"

        for side in ("blue", "red"):
            current_picks = picks[side]
            role_counts = Counter(
                self.champion_roles[champion_id]
                for champion_id in current_picks
                if champion_id in self.champion_roles
            )
            features.extend(role_counts[role] / 5.0 for role in _ROLES)

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
            features.extend(
                sum(values) / len(values) if values else 0.0 for values in profile_values
            )

            side_traits = [self.champion_traits.get(champion_id) for champion_id in current_picks]
            side_traits = [traits for traits in side_traits if traits is not None]
            features.extend(
                sum(tag in traits.tags for traits in side_traits) / 5.0 for tag in self.tag_names
            )
            features.extend(
                sum(traits.info.get(field, 0.0) for traits in side_traits)
                / (10.0 * len(side_traits))
                if side_traits
                else 0.0
                for field in _INFO_FIELDS
            )

            for selected in (current_picks, bans[side]):
                for field in _META_FIELDS:
                    values = [
                        getattr(meta, field)
                        for champion_id in selected
                        if (meta := self._meta_for(patch_key, champion_id)) is not None
                    ]
                    features.append(sum(values) / len(values) if values else 0.0)

        return features
