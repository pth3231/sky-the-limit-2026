"""Seeded legal-move search using the trained model as its fitness function."""

import math
import random
from collections.abc import Sequence
from typing import Protocol

from app.engine.draft_order import DRAFT_ORDER, slot_owner
from app.engine.fearless import DraftContext
from app.engine.models import ActionType, DraftRequest
from app.engine.stats import BlendedStats
from app.model.features import DraftState

DEFAULT_POPULATION_SIZE = 48
DEFAULT_GENERATIONS = 20
DEFAULT_EXHAUSTIVE_THRESHOLD = 256
_TOURNAMENT_SIZE = 3
_ELITE_COUNT = 2
_MUTATION_RATE = 0.15


class Predictor(Protocol):
    """The batch prediction interface required by the search."""

    def predict(self, states: Sequence[DraftState]) -> list[float]:
        """Return blue-side win probabilities for the supplied states."""
        ...


def _available(
    champion_ids: Sequence[str], actions: Sequence[str], unavailable: frozenset[str]
) -> list[str]:
    excluded = unavailable.union(actions)
    return [champion_id for champion_id in champion_ids if champion_id not in excluded]


def _most_likely_reply(
    stats: BlendedStats,
    opponent_team_id: str,
    action: ActionType,
    champion_ids: Sequence[str],
    actions: Sequence[str],
    unavailable: frozenset[str],
) -> str | None:
    legal = _available(champion_ids, actions, unavailable)
    if not legal:
        return None

    team_counts = {
        record.champion_id: record.picks if action == "pick" else record.bans
        for record in stats.team_history(opponent_team_id)
    }
    historically_used = [
        champion_id for champion_id in legal if team_counts.get(champion_id, 0) > 0
    ]
    if historically_used:
        return min(
            historically_used, key=lambda champion_id: (-team_counts[champion_id], champion_id)
        )

    global_counts = {
        champion_id: record.picks if action == "pick" else record.bans
        for champion_id, record in stats.champions.items()
    }
    globally_used = [champion_id for champion_id in legal if global_counts.get(champion_id, 0) > 0]
    if globally_used:
        return min(
            globally_used, key=lambda champion_id: (-global_counts[champion_id], champion_id)
        )
    return legal[0]


def _repair(
    genome: Sequence[str],
    request: DraftRequest,
    context: DraftContext,
    stats: BlendedStats,
    opponent_team_id: str,
    champion_ids: Sequence[str],
    phase_end: int,
    rng: random.Random,
) -> tuple[tuple[str, ...], list[str]] | None:
    actions = list(request.actions)
    genes: list[str] = []
    gene_index = 0
    for index in range(len(request.actions), phase_end):
        slot = DRAFT_ORDER[index]
        if slot.side == request.our_side:
            legal = _available(champion_ids, actions, context.unavailable)
            if not legal:
                return None
            preferred = genome[gene_index] if gene_index < len(genome) else ""
            selected = preferred if preferred in legal else rng.choice(legal)
            genes.append(selected)
            actions.append(selected)
            gene_index += 1
        else:
            reply = _most_likely_reply(
                stats,
                opponent_team_id,
                slot.action,
                champion_ids,
                actions,
                context.unavailable,
            )
            if reply is None:
                return None
            actions.append(reply)
    return tuple(genes), actions


def _enumerate_genomes(
    request: DraftRequest,
    context: DraftContext,
    stats: BlendedStats,
    opponent_team_id: str,
    champion_ids: Sequence[str],
    phase_end: int,
) -> list[tuple[str, ...]]:
    genomes: list[tuple[str, ...]] = []

    def visit(index: int, actions: list[str], genes: list[str]) -> None:
        if index == phase_end:
            genomes.append(tuple(genes))
            return
        slot = DRAFT_ORDER[index]
        if slot.side == request.our_side:
            for champion_id in _available(champion_ids, actions, context.unavailable):
                visit(index + 1, [*actions, champion_id], [*genes, champion_id])
            return
        reply = _most_likely_reply(
            stats,
            opponent_team_id,
            slot.action,
            champion_ids,
            actions,
            context.unavailable,
        )
        if reply is not None:
            visit(index + 1, [*actions, reply], genes)

    visit(len(request.actions), list(request.actions), [])
    return genomes


def _evaluate(
    genomes: Sequence[tuple[str, ...]],
    request: DraftRequest,
    context: DraftContext,
    stats: BlendedStats,
    opponent_team_id: str,
    champion_ids: Sequence[str],
    phase_end: int,
    predictor: Predictor,
    rng: random.Random,
) -> dict[tuple[str, ...], float]:
    repaired: list[tuple[str, ...]] = []
    states: list[DraftState] = []
    patch = request.patch or (stats.patches[0] if stats.patches else "unknown")
    for genome in dict.fromkeys(genomes):
        result = _repair(
            genome,
            request,
            context,
            stats,
            opponent_team_id,
            champion_ids,
            phase_end,
            rng,
        )
        if result is None:
            continue
        repaired_genome, actions = result
        repaired.append(repaired_genome)
        states.append(DraftState(patch=patch, actions=actions))
    if not states:
        return {}

    probabilities = predictor.predict(states)
    if len(probabilities) != len(repaired):
        raise ValueError("Predictor must return one probability per draft state")
    return {
        genome: probability if request.our_side == "blue" else 1.0 - probability
        for genome, probability in zip(repaired, probabilities, strict=True)
    }


def _tournament(
    population: list[tuple[str, ...]],
    scores: dict[tuple[str, ...], float],
    rng: random.Random,
) -> tuple[str, ...]:
    contestants = [rng.choice(population) for _ in range(min(_TOURNAMENT_SIZE, len(population)))]
    return max(contestants, key=lambda genome: scores.get(genome, float("-inf")))


def search_moves(
    request: DraftRequest,
    context: DraftContext,
    stats: BlendedStats,
    opponent_team_id: str,
    champion_ids: Sequence[str],
    predictor: Predictor,
    *,
    result_count: int = 5,
    seed: int = 0,
    population_size: int = DEFAULT_POPULATION_SIZE,
    generations: int = DEFAULT_GENERATIONS,
    exhaustive_threshold: int = DEFAULT_EXHAUSTIVE_THRESHOLD,
) -> list[tuple[str, float]]:
    """Rank distinct legal first moves by model fitness after the current draft phase."""
    slot = context.next_slot
    if slot is None or slot_owner(slot, context.our_side) != "us" or result_count <= 0:
        return []
    if population_size < 1 or generations < 0 or exhaustive_threshold < 0:
        raise ValueError("Search sizes must be non-negative and population_size must be positive")

    start = len(request.actions)
    phase_end = next(
        (
            index
            for index in range(start + 1, len(DRAFT_ORDER))
            if DRAFT_ORDER[index].action != slot.action
        ),
        len(DRAFT_ORDER),
    )
    our_slots = sum(
        DRAFT_ORDER[index].side == request.our_side for index in range(start, phase_end)
    )
    pool = sorted(set(champion_ids))
    available = _available(pool, request.actions, context.unavailable)
    if not available or our_slots == 0:
        return []

    rng = random.Random(seed)
    upper_bound = math.perm(len(available), our_slots) if len(available) >= our_slots else 0
    best_by_move: dict[str, float] = {}
    if upper_bound <= exhaustive_threshold:
        genomes = _enumerate_genomes(
            request,
            context,
            stats,
            opponent_team_id,
            pool,
            phase_end,
        )
        scores = _evaluate(
            genomes,
            request,
            context,
            stats,
            opponent_team_id,
            pool,
            phase_end,
            predictor,
            rng,
        )
        for genome, score in scores.items():
            first_move = genome[0]
            best_by_move[first_move] = max(score, best_by_move.get(first_move, float("-inf")))
    else:
        population: list[tuple[str, ...]] = []
        seen: set[tuple[str, ...]] = set()
        attempts = 0
        while len(population) < population_size and attempts < population_size * 20:
            attempts += 1
            proposal = tuple(rng.choice(available) for _ in range(our_slots))
            result = _repair(
                proposal,
                request,
                context,
                stats,
                opponent_team_id,
                pool,
                phase_end,
                rng,
            )
            if result is not None and result[0] not in seen:
                seen.add(result[0])
                population.append(result[0])
        if not population:
            return []

        for generation in range(generations + 1):
            scores = _evaluate(
                population,
                request,
                context,
                stats,
                opponent_team_id,
                pool,
                phase_end,
                predictor,
                rng,
            )
            for genome, score in scores.items():
                first_move = genome[0]
                best_by_move[first_move] = max(score, best_by_move.get(first_move, float("-inf")))
            ranked_population = sorted(
                scores,
                key=lambda genome: (-scores[genome], genome),
            )
            if generation == generations:
                break

            next_population = ranked_population[:_ELITE_COUNT]
            next_seen = set(next_population)
            attempts = 0
            while len(next_population) < population_size and attempts < population_size * 20:
                attempts += 1
                first = _tournament(population, scores, rng)
                second = _tournament(population, scores, rng)
                child = [
                    first[index] if rng.random() < 0.5 else second[index]
                    for index in range(our_slots)
                ]
                for index in range(our_slots):
                    if rng.random() < _MUTATION_RATE:
                        child[index] = rng.choice(available)
                repaired_child = _repair(
                    child,
                    request,
                    context,
                    stats,
                    opponent_team_id,
                    pool,
                    phase_end,
                    rng,
                )
                if repaired_child is not None and repaired_child[0] not in next_seen:
                    next_seen.add(repaired_child[0])
                    next_population.append(repaired_child[0])
            if len(next_population) < population_size:
                next_population.extend(ranked_population[: population_size - len(next_population)])
            population = next_population
    return sorted(best_by_move.items(), key=lambda item: (-item[1], item[0]))[:result_count]
