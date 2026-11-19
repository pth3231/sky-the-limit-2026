"""Draft recommendation service tying rules, patch blending and scoring together."""

from collections.abc import Mapping
from pathlib import Path

from pydantic import TypeAdapter

from app.engine.draft_order import slot_owner, to_slot_model
from app.engine.errors import DraftStateError
from app.engine.fearless import DraftContext, build_context
from app.engine.models import (
    ActionType,
    Candidate,
    ChampionSummary,
    DraftRequest,
    DraftResponse,
    OpponentTendency,
    TeamSummary,
)
from app.engine.scoring import Scorer, series_win_probability
from app.engine.stats import DEFAULT_MIN_GAMES, BlendedStats, blend_patches
from app.pipeline.aggregates import PatchAggregate, patch_sort_key
from app.pipeline.meta import Champion, Team, load_champions, load_teams

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
    ) -> None:
        self._aggregates = dict(aggregates)
        self._teams = dict(teams)
        self._our_team_id = our_team_id
        self._min_games = min_games
        names = {champion_id: champion.name for champion_id, champion in champions.items()}
        for patch in self._aggregates.values():
            for item in patch.champions:
                names.setdefault(item.champion_id, item.champion_name)
        self._champion_names = names

    def patches(self) -> list[str]:
        """Return patch labels with data, newest first."""
        return sorted(self._aggregates, key=patch_sort_key, reverse=True)

    def champion_list(self) -> list[ChampionSummary]:
        """Return every draftable champion, sorted by name."""
        return sorted(
            (
                ChampionSummary(champion_id=champion_id, name=name)
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

    def recommend(self, request: DraftRequest) -> DraftResponse:
        """Validate the draft state and return a recommendation with its context."""
        opponent = self._teams.get(request.opponent_team_id)
        if opponent is None:
            raise DraftStateError(f"Unknown opponent team: {request.opponent_team_id}")
        context = build_context(request, self._champion_names)
        stats = blend_patches(self._aggregates, request.patch, self._min_games)
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

        game_win_rate = scorer.game_win_probability(context.our_picks, context.opponent_picks)
        wins, losses = context.series_score
        win_rate = series_win_probability(game_win_rate, wins, losses, request.series_format)
        ranked = self._rank(scorer, context)
        recommendation = ranked[0] if ranked else None
        factors = (
            recommendation.contributing_factors
            if recommendation
            else scorer.composition_factors(context.our_picks, context.opponent_picks)
        )
        next_action: ActionType = context.next_slot.action if context.next_slot else "pick"

        return DraftResponse(
            game_number=context.game_number,
            patches_used=list(stats.patches),
            next_slot=(
                to_slot_model(context.next_slot, request.our_side) if context.next_slot else None
            ),
            recommendation=recommendation,
            alternatives=ranked[1:],
            win_rate=round(win_rate, 4),
            game_win_rate=round(game_win_rate, 4),
            opponent_tendencies=self._tendencies(stats, opponent, next_action, context),
            contributing_factors=factors,
            fearless_excluded_ids=sorted(context.fearless_excluded),
            unavailable_ids=sorted(context.unavailable),
            sample_note=self._sample_note(stats, opponent),
        )

    def _rank(self, scorer: Scorer, context: DraftContext) -> list[Candidate]:
        slot = context.next_slot
        if slot is None or slot_owner(slot, context.our_side) != "us":
            return []
        score = scorer.score_pick if slot.action == "pick" else scorer.score_ban
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
                f"Limited data: only {stats.game_count} pro games on patches "
                f"{', '.join(stats.patches)}."
            )
        if stats.team_games(opponent.team_id) == 0:
            notes.append(
                f"No recorded games for {opponent.name} on these patches, "
                "so their tendencies are unknown."
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
    return DraftEngine(
        aggregates,
        load_champions(data_dir),
        teams,
        our_team_id=resolve_our_team_id(teams, our_team_id),
        min_games=min_games,
    )
