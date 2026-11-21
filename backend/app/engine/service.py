"""Draft recommendation service tying rules, patch blending and scoring together."""

import hashlib
from collections.abc import Mapping
from pathlib import Path

from pydantic import TypeAdapter

from app.engine.draft_order import slot_owner, to_slot_model
from app.engine.errors import DraftStateError
from app.engine.fearless import DraftContext, build_context
from app.engine.ga import Predictor, search_moves
from app.engine.models import (
    ActionType,
    Candidate,
    ChainRecommendation,
    ChainStep,
    ChampionSummary,
    DraftRequest,
    DraftResponse,
    OpponentTendency,
    RosterResponse,
    TeamRoster,
    TeamSummary,
)
from app.engine.scoring import Scorer, series_win_probability
from app.engine.stats import DEFAULT_MIN_GAMES, MIN_SUBJECT_GAMES, BlendedStats, blend_patches
from app.model.features import DraftState
from app.model.predictor import WinPredictor
from app.pipeline.aggregates import PatchAggregate, patch_sort_key
from app.pipeline.ddragon import load_portrait_urls, normalize_champion_name
from app.pipeline.meta import Champion, Team, load_champions, load_teams
from app.pipeline.rosters import RosterSnapshot, load_rosters

ALTERNATIVE_COUNT = 4
TENDENCY_COUNT = 5
_AGGREGATES_ADAPTER = TypeAdapter(dict[str, PatchAggregate])


def resolve_our_team_id(teams: Mapping[str, Team], configured: str | None) -> str:
    """Return the configured team ID, or the LoL team whose name starts with "Cloud9"."""
    if configured:
        if configured not in teams:
            raise ValueError(f"Configured team {configured} is not a LoL team in meta/teams.csv")
        return configured
    for team_id in sorted(teams):
        if teams[team_id].name.casefold().startswith("cloud9"):
            return team_id
    raise ValueError("No LoL team named Cloud9 found; set CLOUD9_TEAM_ID")


class DraftEngine:
    """Serves stateless draft recommendations from patch aggregates and metadata."""

    def __init__(
        self,
        aggregates: Mapping[str, PatchAggregate],
        champions: Mapping[str, Champion],
        teams: Mapping[str, Team],
        *,
        our_team_id: str,
        min_games: int = DEFAULT_MIN_GAMES,
        predictor: Predictor | None = None,
        portrait_urls: Mapping[str, str] | None = None,
        rosters: Mapping[str, RosterSnapshot] | None = None,
    ) -> None:
        self._aggregates = dict(aggregates)
        self._teams = dict(teams)
        self._our_team_id = our_team_id
        self._min_games = min_games
        self._predictor = predictor
        self._rosters = dict(rosters or {})
        self._stats_cache: dict[str | None, BlendedStats] = {}
        names = {champion_id: champion.name for champion_id, champion in champions.items()}
        for patch in self._aggregates.values():
            for item in patch.champions:
                names.setdefault(item.champion_id, item.champion_name)
        self._champion_names = names
        self._champion_roles = {
            champion_id: champion.role for champion_id, champion in champions.items()
        }
        available_portraits = portrait_urls or {}
        self._champion_portrait_urls = {
            champion_id: available_portraits.get(normalize_champion_name(champion.name))
            for champion_id, champion in champions.items()
        }

    def patches(self) -> list[str]:
        """Return patch labels with data, newest first."""
        return sorted(self._aggregates, key=patch_sort_key, reverse=True)

    def champion_list(self) -> list[ChampionSummary]:
        """Return every draftable champion, sorted by name."""
        return sorted(
            (
                ChampionSummary(
                    champion_id=champion_id,
                    name=name,
                    role=self._champion_roles.get(champion_id),  # type: ignore[arg-type]
                    portrait_url=self._champion_portrait_urls.get(champion_id),
                )
                for champion_id, name in self._champion_names.items()
            ),
            key=lambda champion: champion.name,
        )

    def opponents(self) -> list[TeamSummary]:
        """Return teams with recorded games other than ours, sorted by name."""
        team_ids = {
            item.team_id
            for patch in self._aggregates.values()
            for item in patch.teams
            if item.team_id != self._our_team_id and item.team_id in self._teams
        }
        return sorted(
            (TeamSummary(team_id=team_id, name=self._teams[team_id].name) for team_id in team_ids),
            key=lambda team: team.name,
        )

    def _opponent(self, team_id: str) -> Team:
        opponent = self._teams.get(team_id)
        if opponent is None:
            raise DraftStateError(f"Unknown opponent team: {team_id}")
        if opponent.team_id == self._our_team_id:
            raise DraftStateError("Your team cannot play itself as the opponent")
        return opponent

    def rosters(self, opponent_team_id: str) -> RosterResponse:
        """Return historical snapshots, preserving identities when metadata is missing."""
        opponent = self._opponent(opponent_team_id)

        def team_roster(team_id: str, name: str) -> TeamRoster:
            snapshot = self._rosters.get(team_id)
            return TeamRoster(
                team_id=team_id,
                name=name,
                as_of=snapshot.as_of if snapshot else None,
                players=snapshot.players if snapshot else [],
            )

        our_team = self._teams.get(self._our_team_id)
        return RosterResponse(
            our_team=team_roster(self._our_team_id, our_team.name if our_team else "Your team"),
            opponent_team=team_roster(opponent.team_id, opponent.name),
        )

    def recommend(self, request: DraftRequest) -> DraftResponse:
        """Validate the draft state and return a recommendation with its context."""
        opponent = self._opponent(request.opponent_team_id)
        context = build_context(request, self._champion_names)
        stats = self._stats_cache.get(request.patch)
        if stats is None:
            stats = blend_patches(self._aggregates, request.patch, self._min_games)
            self._stats_cache[request.patch] = stats
        our_team = self._teams.get(self._our_team_id)
        scorer = Scorer(
            stats,
            champion_names=self._champion_names,
            our_team_id=self._our_team_id,
            our_team_name=our_team.name if our_team else "Your team",
            opponent_team_id=opponent.team_id,
            opponent_team_name=opponent.name,
            our_side=request.our_side,
        )

        if self._predictor is None:
            game_win_rate = scorer.game_win_probability(context.our_picks, context.opponent_picks)
        else:
            patch = request.patch or (stats.patches[0] if stats.patches else "unknown")
            blue_rate = self._predictor.predict([DraftState(patch=patch, actions=request.actions)])[
                0
            ]
            game_win_rate = blue_rate if request.our_side == "blue" else 1.0 - blue_rate
        wins, losses = context.series_score
        win_rate = series_win_probability(game_win_rate, wins, losses, request.series_format)
        ranked = self._rank(request, scorer, context, stats, opponent.team_id)
        recommendation = ranked[0] if ranked else None
        factors = (
            recommendation.contributing_factors
            if recommendation
            else scorer.composition_factors(context.our_picks, context.opponent_picks)
        )
        next_action: ActionType = context.next_slot.action if context.next_slot else "pick"

        chain = self._build_chain(request, scorer, context, stats, opponent.team_id)
        return DraftResponse(
            game_number=context.game_number,
            patches_used=list(stats.all_patches()),
            next_slot=(
                to_slot_model(context.next_slot, request.our_side) if context.next_slot else None
            ),
            recommendation=recommendation,
            alternatives=ranked[1:],
            chain_recommendation=chain,
            win_rate=round(win_rate, 4),
            game_win_rate=round(game_win_rate, 4),
            opponent_tendencies=self._tendencies(stats, opponent, next_action, context),
            contributing_factors=factors,
            fearless_excluded_ids=sorted(context.fearless_excluded),
            unavailable_ids=sorted(context.unavailable),
            sample_note=self._sample_note(stats, opponent),
            engine="model" if self._predictor is not None else "statistical",
        )

    def _build_chain(
        self,
        request: DraftRequest,
        scorer: Scorer,
        context: DraftContext,
        stats: BlendedStats,
        opponent_team_id: str,
    ) -> ChainRecommendation | None:
        """Build a chain of recommended ban/pick moves for the remaining draft slots."""
        from app.engine.draft_order import DRAFT_ORDER, slot_owner

        steps: list[ChainStep] = []
        actions = list(request.actions)
        unavailable = set(context.unavailable)

        for slot_index in range(len(request.actions), len(DRAFT_ORDER)):
            slot = DRAFT_ORDER[slot_index]
            if slot_owner(slot, request.our_side) != "us":
                # Simulate opponent's move using historical tendencies
                legal = [
                    cid
                    for cid in self._champion_names
                    if cid not in unavailable and cid not in actions
                ]
                if not legal:
                    break
                # Pick opponent's most likely champion
                team_counts = {
                    record.champion_id: record.picks if slot.action == "pick" else record.bans
                    for record in stats.team_history(opponent_team_id)
                }
                preferred = sorted(
                    [cid for cid in legal if team_counts.get(cid, 0) > 0],
                    key=lambda cid: -team_counts[cid],
                )
                opponent_pick = preferred[0] if preferred else legal[0]
                actions.append(opponent_pick)
                unavailable.add(opponent_pick)
                continue

            # Our move - score all available champions
            legal = [
                cid for cid in self._champion_names if cid not in unavailable and cid not in actions
            ]
            if not legal:
                break

            score_fn = scorer.score_pick if slot.action == "pick" else scorer.score_ban
            candidates = [score_fn(cid, context.our_picks, context.opponent_picks) for cid in legal]
            candidates.sort(key=lambda c: -c.score)
            best = candidates[0]

            steps.append(
                ChainStep(
                    slot_index=slot_index,
                    action=slot.action,
                    champion_id=best.champion_id,
                    champion_name=best.champion_name,
                    score=round(best.score, 4),
                    reason=best.reason,
                )
            )
            actions.append(best.champion_id)
            unavailable.add(best.champion_id)

        if not steps:
            return None

        # Estimate final win rate using the last candidate's score as proxy
        estimated_win_rate = round(
            scorer.game_win_probability(context.our_picks, context.opponent_picks), 4
        )

        return ChainRecommendation(
            steps=steps,
            estimated_win_rate=estimated_win_rate,
            engine="model" if self._predictor is not None else "statistical",
        )

    @staticmethod
    def _request_seed(request: DraftRequest) -> int:
        payload = request.model_dump_json()
        return int.from_bytes(hashlib.sha256(payload.encode("utf-8")).digest()[:4], "big")

    def _rank(
        self,
        request: DraftRequest,
        scorer: Scorer,
        context: DraftContext,
        stats: BlendedStats,
        opponent_team_id: str,
    ) -> list[Candidate]:
        slot = context.next_slot
        if slot is None or slot_owner(slot, context.our_side) != "us":
            return []
        score = scorer.score_pick if slot.action == "pick" else scorer.score_ban
        if self._predictor is not None:
            moves = search_moves(
                request,
                context,
                stats,
                opponent_team_id,
                tuple(self._champion_names),
                self._predictor,
                result_count=1 + ALTERNATIVE_COUNT,
                seed=self._request_seed(request),
            )
            candidates: list[Candidate] = []
            for champion_id, probability in moves:
                statistical = score(champion_id, context.our_picks, context.opponent_picks)
                signal = (
                    statistical.contributing_factors[0].label
                    if statistical.contributing_factors
                    else "No historical statistical signal is available"
                )
                percent = round(probability * 100)
                candidates.append(
                    statistical.model_copy(
                        update={
                            "score": round(probability, 4),
                            "reason": (
                                f"Model estimates a {percent}% game win chance after this phase. "
                                f"Statistical signal: {signal}."
                            ),
                        }
                    )
                )
            return candidates

        candidates = [
            score(champion_id, context.our_picks, context.opponent_picks)
            for champion_id in self._champion_names
            if champion_id not in context.unavailable
        ]
        candidates.sort(key=lambda candidate: (-candidate.score, candidate.champion_name))
        return candidates[: 1 + ALTERNATIVE_COUNT]

    def _tendencies(
        self,
        stats: BlendedStats,
        opponent: Team,
        action: ActionType,
        context: DraftContext,
    ) -> list[OpponentTendency]:
        team_games = stats.team_games(opponent.team_id)
        if team_games == 0:
            return []
        verb = "picked" if action == "pick" else "banned"
        counted = [
            (record.picks if action == "pick" else record.bans, record.champion_id)
            for record in stats.team_history(opponent.team_id)
            if record.champion_id not in context.unavailable
        ]
        counted = [(count, champion_id) for count, champion_id in counted if count > 0]
        counted.sort(key=lambda item: (-item[0], self._champion_names.get(item[1], item[1])))
        return [
            OpponentTendency(
                champion_id=champion_id,
                champion_name=self._champion_names.get(champion_id, champion_id),
                action=action,
                count=count,
                rate=round(count / team_games, 4),
                label=f"{opponent.name} {verb} it in {count} of {team_games} games",
            )
            for count, champion_id in counted[:TENDENCY_COUNT]
        ]

    def _sample_note(self, stats: BlendedStats, opponent: Team) -> str | None:
        notes: list[str] = []
        if stats.game_count < self._min_games:
            notes.append(
                f"Limited global sample: only {stats.game_count} pro games on patches "
                f"{', '.join(stats.patches)}."
            )
        if stats.team_games(opponent.team_id) == 0:
            notes.append(
                f"No recorded games for {opponent.name} on these patches, "
                "so their tendencies are unknown."
            )
        elif stats.team_games(opponent.team_id) < MIN_SUBJECT_GAMES:
            games = stats.team_games(opponent.team_id)
            notes.append(
                f"Limited opponent history: only {games} recorded game"
                f"{'s' if games != 1 else ''} for {opponent.name}; tendencies may be unreliable."
            )
        return " ".join(notes) or None


def load_engine(
    derived_dir: Path,
    data_dir: Path,
    our_team_id: str | None,
    min_games: int = DEFAULT_MIN_GAMES,
) -> DraftEngine:
    """Load ``aggregates.json`` and LoL metadata into a ready ``DraftEngine``."""
    aggregates_path = derived_dir / "aggregates.json"
    if not aggregates_path.is_file():
        raise FileNotFoundError(
            f"{aggregates_path} not found; run `uv run python -m app.pipeline.build` first"
        )
    aggregates = _AGGREGATES_ADAPTER.validate_json(aggregates_path.read_bytes())
    teams = load_teams(data_dir)
    champions = load_champions(data_dir)
    predictor = None
    model_path = derived_dir / "model.pt"
    if model_path.is_file():
        try:
            predictor = WinPredictor.load(model_path)
        except (ImportError, OSError, RuntimeError, TypeError, ValueError):
            predictor = None
    return DraftEngine(
        aggregates,
        champions,
        teams,
        our_team_id=resolve_our_team_id(teams, our_team_id),
        min_games=min_games,
        predictor=predictor,
        portrait_urls=load_portrait_urls(derived_dir / "ddragon.json"),
        rosters=load_rosters(data_dir),
    )
