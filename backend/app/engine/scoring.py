"""Explainable statistical scoring of draft candidates and win probabilities.

Every edge is a smoothed win rate minus its baseline, in win-rate points. A team's strength is
the sum of its champions' meta edges, the team's comfort edges and same-team pair synergy edges,
each capped so no single signal dominates. The signals overlap (a strong champion also lifts its
pairs and its team's record), so the game estimate damps the summed gap through a logistic link
with slope 1.2: about ``0.5 + 0.3 * gap`` near an even game, and rarely beyond 15-85%.
"""

import math
from collections.abc import Mapping, Sequence
from functools import cache
from itertools import combinations
from typing import Literal

from pydantic import BaseModel, ConfigDict

from app.engine.fearless import wins_needed
from app.engine.models import Candidate, ContributingFactor, SeriesFormat, Side, Signal
from app.engine.stats import BlendedStats

PRIOR_RATE = 0.5
PRIOR_GAMES = 10
SIDE_PRIOR_GAMES = 20
META_CAP = 0.10
COMFORT_CAP = 0.05
SYNERGY_CAP = 0.05
SIDE_CAP = 0.05
MATCHUP_CAP = 0.05
LOGISTIC_SLOPE = 1.2
STRONG_SIGNAL_GAMES = 30
SOME_SIGNAL_GAMES = 10

_EdgeKind = Literal["meta", "comfort", "synergy", "matchup"]


def smoothed_rate(
    wins: int,
    games: int,
    prior: float = PRIOR_RATE,
    prior_games: int = PRIOR_GAMES,
) -> float:
    """Return a win rate pulled toward ``prior`` as if ``prior_games`` extra games were played."""
    return (wins + prior * prior_games) / (games + prior_games)


def signal_for(games: int) -> Signal:
    """Return the plain-language strength of a signal backed by ``games`` games."""
    if games >= STRONG_SIGNAL_GAMES:
        return "strong"
    if games >= SOME_SIGNAL_GAMES:
        return "some"
    return "limited"


def series_win_probability(
    game_probability: float,
    wins: int,
    losses: int,
    series_format: SeriesFormat,
) -> float:
    """Return the chance to win the series from the current score.

    The current game uses ``game_probability``; later games are treated as even.
    """
    needed = wins_needed(series_format)

    @cache
    def chance(won: int, lost: int, probability: float) -> float:
        if won >= needed:
            return 1.0
        if lost >= needed:
            return 0.0
        return probability * chance(won + 1, lost, PRIOR_RATE) + (1 - probability) * chance(
            won, lost + 1, PRIOR_RATE
        )

    return chance(wins, losses, game_probability)


def _clamp(value: float, cap: float) -> float:
    return max(-cap, min(cap, value))


def _percent(wins: int, games: int) -> int:
    return round(100 * wins / games) if games else 0


def _logistic(edge: float) -> float:
    return 1 / (1 + math.exp(-LOGISTIC_SLOPE * edge))


def _edge_impact(edge: float) -> float:
    """Approximate a strength edge as a win-probability change from an even baseline."""
    return _logistic(edge) - PRIOR_RATE


class _Edge(BaseModel):
    model_config = ConfigDict(frozen=True)
    kind: _EdgeKind
    champion_id: str
    edge: float
    wins: int
    games: int
    partner_id: str | None = None


class Scorer:
    """Scores candidates and compositions for one matchup on the blended patches."""

    def __init__(
        self,
        stats: BlendedStats,
        *,
        champion_names: Mapping[str, str],
        our_team_id: str,
        our_team_name: str,
        opponent_team_id: str,
        opponent_team_name: str,
        our_side: Side,
        champion_roles: Mapping[str, str | None],
    ) -> None:
        self._stats = stats
        self._names = champion_names
        self._our_team_id = our_team_id
        self._our_team_name = our_team_name
        self._opponent_team_id = opponent_team_id
        self._opponent_team_name = opponent_team_name
        self._our_side: Side = our_side
        self._champion_roles = champion_roles

    def _name(self, champion_id: str) -> str:
        return self._names.get(champion_id, champion_id)

    def _champion_rate(self, champion_id: str) -> float:
        record = self._stats.champion(champion_id)
        return smoothed_rate(record.wins, record.decided_games) if record else PRIOR_RATE

    def _meta_edge(self, champion_id: str) -> _Edge:
        record = self._stats.champion(champion_id)
        wins, games = (record.wins, record.decided_games) if record else (0, 0)
        edge = _clamp(self._champion_rate(champion_id) - PRIOR_RATE, META_CAP)
        return _Edge(kind="meta", champion_id=champion_id, edge=edge, wins=wins, games=games)

    def _comfort_edge(self, team_id: str, champion_id: str) -> _Edge | None:
        record = self._stats.team_champion(team_id, champion_id)
        if record is None or record.decided_picks == 0:
            return None
        champion_rate = self._champion_rate(champion_id)
        team_rate = smoothed_rate(record.wins, record.decided_picks, prior=champion_rate)
        edge = _clamp(team_rate - champion_rate, COMFORT_CAP)
        return _Edge(
            kind="comfort",
            champion_id=champion_id,
            edge=edge,
            wins=record.wins,
            games=record.decided_picks,
        )

    def _synergy_edge(self, champion_id: str, partner_id: str) -> _Edge | None:
        record = self._stats.pair(champion_id, partner_id)
        if record is None or record.games == 0:
            return None
        expected = (self._champion_rate(champion_id) + self._champion_rate(partner_id)) / 2
        edge = _clamp(
            smoothed_rate(record.wins, record.games, prior=expected) - expected, SYNERGY_CAP
        )
        return _Edge(
            kind="synergy",
            champion_id=champion_id,
            edge=edge,
            wins=record.wins,
            games=record.games,
            partner_id=partner_id,
        )

    def _matchup_edge(self, champion_id: str, opponent_picks: Sequence[str]) -> _Edge | None:
        role = self._champion_roles.get(champion_id)
        if role is None:
            return None
        opponent_id = next(
            (pick for pick in opponent_picks if self._champion_roles.get(pick) == role), None
        )
        if opponent_id is None:
            return None
        record = self._stats.matchup(role, champion_id, opponent_id)
        if record is None or record.games == 0:
            return None
        expected = self._champion_rate(champion_id)
        edge = _clamp(
            smoothed_rate(record.wins, record.games, prior=expected) - expected, MATCHUP_CAP
        )
        return _Edge(
            kind="matchup",
            champion_id=champion_id,
            edge=edge,
            wins=record.wins,
            games=record.games,
            partner_id=opponent_id,
        )

    def _candidate_edges(
        self,
        team_id: str,
        champion_id: str,
        teammates: Sequence[str],
        opponents: Sequence[str],
    ) -> list[_Edge]:
        edges = [self._meta_edge(champion_id)]
        comfort = self._comfort_edge(team_id, champion_id)
        if comfort:
            edges.append(comfort)
        for partner_id in teammates:
            synergy = self._synergy_edge(champion_id, partner_id)
            if synergy:
                edges.append(synergy)
        matchup = self._matchup_edge(champion_id, opponents)
        if matchup:
            edges.append(matchup)
        return edges

    def _team_edges(
        self, team_id: str, picks: Sequence[str], opponents: Sequence[str]
    ) -> list[_Edge]:
        edges: list[_Edge] = []
        for champion_id in picks:
            edges.append(self._meta_edge(champion_id))
            comfort = self._comfort_edge(team_id, champion_id)
            if comfort:
                edges.append(comfort)
            matchup = self._matchup_edge(champion_id, opponents)
            if matchup:
                edges.append(matchup)
        for champion_a_id, champion_b_id in combinations(picks, 2):
            synergy = self._synergy_edge(champion_a_id, champion_b_id)
            if synergy:
                edges.append(synergy)
        return edges

    def _side_edge(self) -> tuple[float, int, int]:
        wins, games = self._stats.side_record(self._our_side)
        edge = _clamp(
            smoothed_rate(wins, games, prior_games=SIDE_PRIOR_GAMES) - PRIOR_RATE, SIDE_CAP
        )
        return edge, wins, games

    def _strength_gap(self, our_picks: Sequence[str], opponent_picks: Sequence[str]) -> float:
        ours = sum(
            edge.edge for edge in self._team_edges(self._our_team_id, our_picks, opponent_picks)
        )
        theirs = sum(
            edge.edge
            for edge in self._team_edges(self._opponent_team_id, opponent_picks, our_picks)
        )
        return self._side_edge()[0] + ours - theirs

    def game_win_probability(
        self,
        our_picks: Sequence[str],
        opponent_picks: Sequence[str],
    ) -> float:
        """Return our estimated chance to win the current game with these picks."""
        return _logistic(self._strength_gap(our_picks, opponent_picks))

    def _label(self, edge: _Edge, owner: Literal["candidate", "ours", "theirs"]) -> str:
        team_name = self._opponent_team_name if owner == "theirs" else self._our_team_name
        name = self._name(edge.champion_id)
        if edge.kind == "meta":
            if edge.games == 0:
                return "No recent pro games on these patches"
            rate = f"{_percent(edge.wins, edge.games)}% of pro games on recent patches"
            subject = {
                "candidate": "Wins",
                "ours": f"Your {name} wins",
                "theirs": f"Their {name} wins",
            }
            return f"{subject[owner]} {rate} ({edge.games} games)"
        if edge.kind == "comfort":
            target = "it" if owner == "candidate" else name
            return f"{team_name} won {edge.wins} of {edge.games} games with {target}"
        if edge.kind == "matchup":
            opponent_name = self._name(edge.partner_id or "")
            verb = "beats" if edge.edge >= 0 else "loses to"
            subject = {
                "candidate": f"Lane matchup {verb} their {opponent_name}",
                "ours": f"Your {name} {verb} their {opponent_name} in lane",
                "theirs": f"Their {name} {verb} your {opponent_name} in lane",
            }
            return f"{subject[owner]} (won {edge.wins} of {edge.games})"
        partner = self._name(edge.partner_id or "")
        quality = "Strong" if edge.edge >= 0 else "Awkward"
        if owner == "candidate":
            return f"{quality} with your {partner} (won {edge.wins} of {edge.games} together)"
        record = f"won {edge.wins} of {edge.games}"
        return f"{quality} pair for {team_name}: {name} and {partner} ({record})"

    def _factor(
        self,
        edge: _Edge,
        owner: Literal["candidate", "ours", "theirs"],
        impact: float,
    ) -> ContributingFactor:
        return ContributingFactor(
            label=self._label(edge, owner),
            impact=round(impact, 3),
            signal=signal_for(edge.games),
        )

    @staticmethod
    def _ordered(factors: list[ContributingFactor]) -> list[ContributingFactor]:
        return sorted(factors, key=lambda factor: abs(factor.impact), reverse=True)

    def score_pick(
        self,
        champion_id: str,
        our_picks: Sequence[str],
        opponent_picks: Sequence[str],
    ) -> Candidate:
        """Score picking ``champion_id`` for us next."""
        probability = self.game_win_probability([*our_picks, champion_id], opponent_picks)
        edges = self._candidate_edges(self._our_team_id, champion_id, our_picks, opponent_picks)
        factors = self._ordered(
            [self._factor(edge, "candidate", _edge_impact(edge.edge)) for edge in edges]
        )
        current = self.game_win_probability(our_picks, opponent_picks)
        percent = round(100 * probability)
        if probability > current:
            headline = f"Raises your estimated game win chance to {percent}%"
        elif probability < current:
            headline = f"Lowers your estimated game win chance to {percent}%"
        else:
            headline = f"Keeps your estimated game win chance unchanged at {percent}%"
        return Candidate(
            champion_id=champion_id,
            champion_name=self._name(champion_id),
            score=round(probability, 4),
            reason=f"{headline}. {factors[0].label}.",
            contributing_factors=factors,
        )

    def _opponent_pick_share(self, champion_id: str) -> tuple[int, int]:
        record = self._stats.team_champion(self._opponent_team_id, champion_id)
        return (record.picks if record else 0), self._stats.team_games(self._opponent_team_id)

    def score_ban(
        self,
        champion_id: str,
        our_picks: Sequence[str],
        opponent_picks: Sequence[str],
    ) -> Candidate:
        """Score banning ``champion_id`` so the opponent cannot pick it."""
        current = self.game_win_probability(our_picks, opponent_picks)
        if_they_pick = self.game_win_probability(our_picks, [*opponent_picks, champion_id])
        threat = current - if_they_pick
        picks, team_games = self._opponent_pick_share(champion_id)
        share = picks / team_games if team_games else 0.0
        weight = 0.5 + 0.5 * share
        score = current + threat * weight

        edges = self._candidate_edges(
            self._opponent_team_id, champion_id, opponent_picks, our_picks
        )
        factors = [self._factor(edge, "candidate", _edge_impact(edge.edge)) for edge in edges]
        for index, edge in enumerate(edges):
            if edge.kind == "comfort":
                label = f"{self._opponent_team_name} won {edge.wins} of {edge.games} games with it"
                factors[index] = factors[index].model_copy(update={"label": label})
            elif edge.kind == "synergy":
                partner = self._name(edge.partner_id or "")
                label = f"Pairs with their {partner} (won {edge.wins} of {edge.games} together)"
                factors[index] = factors[index].model_copy(update={"label": label})
            elif edge.kind == "matchup":
                partner = self._name(edge.partner_id or "")
                verb = "beats" if edge.edge >= 0 else "loses to"
                label = f"Lane matchup {verb} your {partner} (won {edge.wins} of {edge.games})"
                factors[index] = factors[index].model_copy(update={"label": label})
        if picks:
            factors.append(
                ContributingFactor(
                    label=f"{self._opponent_team_name} picked it in {picks} of {team_games} games",
                    impact=round(threat * 0.5 * share, 3),
                    signal=signal_for(team_games),
                )
            )
        factors = self._ordered(factors)

        points = round(100 * threat)
        if points > 0:
            headline = (
                f"Takes away a champion worth about {points} points of win chance "
                f"to {self._opponent_team_name}"
            )
        else:
            headline = f"Low-impact ban: {self._opponent_team_name} gains little from it"
        return Candidate(
            champion_id=champion_id,
            champion_name=self._name(champion_id),
            score=round(score, 4),
            reason=f"{headline}. {factors[0].label}.",
            contributing_factors=factors,
        )

    def composition_factors(
        self,
        our_picks: Sequence[str],
        opponent_picks: Sequence[str],
    ) -> list[ContributingFactor]:
        """Explain the current game estimate: side, our picks, and the opponent's picks."""
        side_edge, side_wins, side_games = self._side_edge()
        side_label = (
            "No decided pro games on these patches"
            if side_games == 0
            else f"{self._our_side.capitalize()} side won {_percent(side_wins, side_games)}% "
            f"of recent pro games ({side_games} games)"
        )
        factors = [
            ContributingFactor(
                label=side_label,
                impact=round(_edge_impact(side_edge), 3),
                signal=signal_for(side_games),
            )
        ]
        factors.extend(
            self._factor(edge, "ours", _edge_impact(edge.edge))
            for edge in self._team_edges(self._our_team_id, our_picks, opponent_picks)
        )
        factors.extend(
            self._factor(edge, "theirs", -_edge_impact(edge.edge))
            for edge in self._team_edges(self._opponent_team_id, opponent_picks, our_picks)
        )
        return self._ordered(factors)
