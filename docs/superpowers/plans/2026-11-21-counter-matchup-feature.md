# Counter-Matchup Feature Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a per-role champion-vs-champion counter-matchup signal (item 1 of Milestone 1 in `docs/superpowers/specs/2026-11-21-recommendation-model-improvements-design.md`) to both the explainable statistical scorer and the NN's feature vector, so recommendations account for "how does my pick do against their pick in the same role" — a signal currently missing entirely.

**Architecture:** Compute per-role opposing-pick win counts during the existing `compute_aggregates()` pass (pipeline layer), sum them across blended patches in `BlendedStats` (engine layer) for the statistical `Scorer`, and independently fit the same signal from training games inside `FeatureSpace.fit()` (model layer) for the NN — mirroring the project's existing pattern where pipeline aggregates and NN features are computed separately from the same underlying `GameDraft`/`PlayerLine` data.

**Tech Stack:** Python 3.12, Pydantic v2, pytest, existing `backend/app/pipeline`, `backend/app/engine`, `backend/app/model` packages.

## Global Constraints

- Line length 100 chars (Ruff), `ruff check .` / `ruff format --check .` / `basedpyright` / `pytest -q` must all stay green (from `backend/CONVENTIONS.md`).
- TDD mandatory: failing test first, then minimal implementation (`backend/CONVENTIONS.md`).
- Every task requires a GitHub issue before work begins; branch name is `<type>/<issue-number>-<slug>`; PR description must include `Closes #<issue-number>` and the final commit must carry a `Closes: #<N>` trailer (`CONVENTIONS.md`).
- Stage files explicitly (`git add <path>`), never `git add -A`/`git add .`; confirm `git diff --cached --name-only` has no data-derived files before committing (`AGENTS.md`).
- Role values are the plain strings `"top"`, `"jungle"`, `"mid"`, `"bot"`, `"support"` — `app/pipeline/drafts.py` and `app/pipeline/meta.py` each define their own `Role`/`str` typing for this value set independently; this plan keeps new pipeline/engine code on plain `str` (matching `Champion.role: str | None`) and only uses the stricter `Role` Literal inside `app/model/features.py`, matching that file's existing convention.

---

### Task 1: Per-role matchup counts in `compute_aggregates`

**Files:**
- Modify: `backend/app/pipeline/aggregates.py`
- Test: `backend/tests/pipeline/test_aggregates.py`

**Interfaces:**
- Produces: `ChampionMatchupAggregate` (fields `role: str`, `champion_id: str`, `opponent_champion_id: str`, `games: int`, `wins: int`) and `PatchAggregate.matchups: list[ChampionMatchupAggregate]`, consumed by Task 2.

- [ ] **Step 1: Create the GitHub issue and branch**

```bash
gh issue create \
  --title "[Engine] Add per-role counter-matchup signal" \
  --label feature \
  --body "Part of Milestone 1 in docs/superpowers/specs/2026-11-21-recommendation-model-improvements-design.md: add a per-role champion-vs-champion win-rate signal to the statistical scorer and NN features. Currently the scoring/features only know same-team synergy, never how a pick fares against the specific opposing pick in its role."
# note the returned issue number as <N>
git checkout main && git pull --ff-only
git checkout -b feat/<N>-counter-matchup-signal
```

- [ ] **Step 2: Write the failing test**

Add to `backend/tests/pipeline/test_aggregates.py` (add `Callable` to the existing `typing`/`collections.abc` imports at the top of the file, and add `from app.pipeline.drafts import PlayerLine`):

```python
def test_computes_per_role_matchup_counts_from_lineups(make_game: Callable[..., GameDraft]) -> None:
    def lineup(team: str, role: str, champion: str) -> PlayerLine:
        return PlayerLine(team_id=team, role=role, champion_id=champion)

    blue_win = make_game(
        blue="blue", red="red", winner="blue", blue_picks=["A"], red_picks=["C"]
    ).model_copy(update={"lineups": [lineup("blue", "top", "A"), lineup("red", "top", "C")]})
    red_win = make_game(
        blue="blue", red="red", winner="red", blue_picks=["A"], red_picks=["C"]
    ).model_copy(update={"lineups": [lineup("blue", "top", "A"), lineup("red", "top", "C")]})
    missing_role = make_game(blue="blue", red="red", winner="blue", blue_picks=["A"]).model_copy(
        update={"lineups": [lineup("blue", "top", "A")]}
    )

    aggregates = compute_aggregates([blue_win, red_win, missing_role])
    matchups = {
        (item.role, item.champion_id, item.opponent_champion_id): item
        for item in aggregates["16.1"].matchups
    }

    assert (matchups["top", "A", "C"].games, matchups["top", "A", "C"].wins) == (2, 1)
    assert (matchups["top", "C", "A"].games, matchups["top", "C", "A"].wins) == (2, 1)
```

- [ ] **Step 3: Run test to verify it fails**

Run: `cd backend && uv run pytest tests/pipeline/test_aggregates.py::test_computes_per_role_matchup_counts_from_lineups -v`
Expected: FAIL with `KeyError` or `AttributeError` (`PatchAggregate` has no `matchups` yet).

- [ ] **Step 4: Add `ChampionMatchupAggregate` and compute matchup counts**

In `backend/app/pipeline/aggregates.py`, add the model after `class ChampionPairAggregate(BaseModel): ...`:

```python
class ChampionMatchupAggregate(BaseModel):
    """One champion's per-role win count against a specific opposing champion, in decided games."""

    model_config = ConfigDict(frozen=True)

    role: str
    champion_id: str
    opponent_champion_id: str
    games: int
    wins: int
```

Add `matchups: list[ChampionMatchupAggregate]` as a field on `PatchAggregate`, after `team_champions: list[TeamChampionAggregate]`:

```python
    team_champions: list[TeamChampionAggregate]
    teams: list[TeamAggregate]
    matchups: list[ChampionMatchupAggregate]
```

Add an internal counts model next to `_PairCounts`:

```python
class _MatchupCounts(BaseModel):
    games: int = 0
    wins: int = 0
```

In `compute_aggregates`, add `matchups: dict[tuple[str, str, str], _MatchupCounts] = {}` next to the other per-patch dicts (`champions`, `synergies`, `team_champions`), and inside the `for game in games:` loop, directly after the existing synergy block (`for team_id, picked_champions in picked_by_team.items(): ...`), still inside the `if not decided: continue` guard's surviving branch, add:

```python
            lineup_by_role: dict[str, dict[str, str]] = defaultdict(dict)
            for player in game.lineups:
                if player.role is None:
                    continue
                lineup_by_role[player.role][player.team_id] = player.champion_id
            for role, by_team in lineup_by_role.items():
                blue_champion = by_team.get(game.blue_team_id)
                red_champion = by_team.get(game.red_team_id)
                if blue_champion is None or red_champion is None:
                    continue
                blue_won = game.winner_team_id == game.blue_team_id
                forward = matchups.setdefault((role, blue_champion, red_champion), _MatchupCounts())
                forward.games += 1
                forward.wins += blue_won
                backward = matchups.setdefault((role, red_champion, blue_champion), _MatchupCounts())
                backward.games += 1
                backward.wins += not blue_won
```

After the `for game in games:` loop, build the results list next to `synergy_results`/`team_results`:

```python
        matchup_results = [
            ChampionMatchupAggregate(
                role=key[0],
                champion_id=key[1],
                opponent_champion_id=key[2],
                games=counts.games,
                wins=counts.wins,
            )
            for key, counts in sorted(matchups.items())
        ]
```

Add `matchups=matchup_results` to the `PatchAggregate(...)` construction at the end of the function.

- [ ] **Step 5: Run test to verify it passes**

Run: `cd backend && uv run pytest tests/pipeline/test_aggregates.py -v`
Expected: PASS (all tests in the file, including the existing ones — confirms the new field didn't break anything).

- [ ] **Step 6: Commit**

```bash
git add backend/app/pipeline/aggregates.py backend/tests/pipeline/test_aggregates.py
git commit -m "feat: compute per-role champion matchup counts in aggregates"
```

---

### Task 2: Blend matchup counts into `BlendedStats`

**Files:**
- Modify: `backend/app/engine/stats.py`
- Test: `backend/tests/engine/test_stats.py`

**Interfaces:**
- Consumes: `ChampionMatchupAggregate`, `PatchAggregate.matchups` from Task 1.
- Produces: `BlendedStats.matchup(role: str, champion_id: str, opponent_champion_id: str) -> ChampionMatchupAggregate | None`, consumed by Task 3.

- [ ] **Step 1: Write the failing test**

Add to `backend/tests/engine/test_stats.py`:

```python
def test_sums_matchup_counts_across_blended_patches(aggregates: dict[str, PatchAggregate]) -> None:
    stats = blend_patches(aggregates, None, min_games=6)

    assert stats.matchup("top", "missing", "missing") is None
```

(This only proves the method exists and returns `None` safely — Task 3's tests exercise real matchup data end-to-end via `Scorer`, since the shared `aggregates` fixture here has no lineups.)

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && uv run pytest tests/engine/test_stats.py::test_sums_matchup_counts_across_blended_patches -v`
Expected: FAIL with `AttributeError: 'BlendedStats' object has no attribute 'matchup'`.

- [ ] **Step 3: Add matchup support to `BlendedStats`**

In `backend/app/engine/stats.py`, add `ChampionMatchupAggregate` to the import from `app.pipeline.aggregates`:

```python
from app.pipeline.aggregates import (
    ChampionAggregate,
    ChampionMatchupAggregate,
    ChampionPairAggregate,
    PatchAggregate,
    TeamAggregate,
    TeamChampionAggregate,
    patch_sort_key,
)
```

Add a field to `BlendedStats`, after `pairs`:

```python
    pairs: Mapping[tuple[str, str], ChampionPairAggregate]
    matchups: Mapping[tuple[str, str, str], ChampionMatchupAggregate]
```

Add an accessor method, after `pair`:

```python
    def matchup(
        self, role: str, champion_id: str, opponent_champion_id: str
    ) -> ChampionMatchupAggregate | None:
        """Return a champion's per-role record against a specific opposing champion."""
        return self.matchups.get((role, champion_id, opponent_champion_id))
```

Add a summing helper, after `_sum_pairs`:

```python
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
```

In `blend_patches`, add `matchups=_sum_matchups(selected),` to the returned `BlendedStats(...)` call.

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && uv run pytest tests/engine/test_stats.py -v`
Expected: PASS (all tests in the file).

- [ ] **Step 5: Commit**

```bash
git add backend/app/engine/stats.py backend/tests/engine/test_stats.py
git commit -m "feat: sum per-role matchup counts into BlendedStats"
```

---

### Task 3: Counter-matchup edge in the statistical `Scorer`

**Files:**
- Modify: `backend/app/engine/scoring.py`
- Modify: `backend/app/engine/service.py:156-164`
- Modify: `backend/app/model/train.py` (the `_scorer_probabilities` function)
- Test: `backend/tests/engine/test_scoring.py`

**Interfaces:**
- Consumes: `BlendedStats.matchup(...)` from Task 2.
- Produces: a `"matchup"`-kind `_Edge` surfaced in `Candidate.contributing_factors` for `score_pick`, `score_ban`, and `composition_factors`. `Scorer.__init__` gains a required keyword-only `champion_roles: Mapping[str, str | None]` parameter — every existing `Scorer(...)` call site must pass it.

- [ ] **Step 1: Write the failing tests**

Add to `backend/tests/engine/test_scoring.py` (uses the existing `NAMES`, `make_game`, `Callable`, `blend_patches`, `compute_aggregates`, `GameDraft` imports already in the file; add `from app.pipeline.drafts import PlayerLine` to its imports):

```python
@pytest.fixture
def matchup_scorer(make_game: Callable[..., GameDraft]) -> Scorer:
    def lineup(team: str, role: str, champion: str) -> PlayerLine:
        return PlayerLine(team_id=team, role=role, champion_id=champion)

    def game(winner: str) -> GameDraft:
        return make_game(
            blue="c9", red="opp", winner=winner, blue_picks=["A"], red_picks=["X"]
        ).model_copy(update={"lineups": [lineup("c9", "top", "A"), lineup("opp", "top", "X")]})

    games = [game("blue") for _ in range(8)] + [game("red") for _ in range(2)]
    stats = blend_patches(compute_aggregates(games), None, min_games=1)
    return Scorer(
        stats,
        champion_names={"A": "Ahri", "X": "Xin Zhao"},
        our_team_id="c9",
        our_team_name="Cloud9",
        opponent_team_id="opp",
        opponent_team_name="Rivals",
        our_side="blue",
        champion_roles={"A": "top", "X": "top"},
    )


def test_matchup_factor_appears_when_opponent_has_same_role_pick(matchup_scorer: Scorer) -> None:
    candidate = matchup_scorer.score_pick("A", our_picks=(), opponent_picks=("X",))

    labels = [factor.label for factor in candidate.contributing_factors]
    assert any("beats their Xin Zhao" in label for label in labels)


def test_matchup_factor_absent_without_a_same_role_opposing_pick(matchup_scorer: Scorer) -> None:
    candidate = matchup_scorer.score_pick("A", our_picks=(), opponent_picks=())

    labels = [factor.label for factor in candidate.contributing_factors]
    assert not any("lane matchup" in label.lower() for label in labels)


def test_ban_candidate_surfaces_matchup_threat_against_our_pick(matchup_scorer: Scorer) -> None:
    candidate = matchup_scorer.score_ban("X", our_picks=("A",), opponent_picks=())

    labels = [factor.label for factor in candidate.contributing_factors]
    assert any("loses to your Ahri" in label for label in labels)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd backend && uv run pytest tests/engine/test_scoring.py -v`
Expected: FAIL — `TypeError: Scorer.__init__() got an unexpected keyword argument 'champion_roles'` (only the three new tests fail here; the existing `scorer` fixture doesn't pass `champion_roles` yet, so it's unaffected until Step 3 makes the parameter required — at which point it would fail until Step 4 fixes it).

- [ ] **Step 3: Add the matchup edge to `Scorer`**

In `backend/app/engine/scoring.py`, add `MATCHUP_CAP = 0.05` next to the other cap constants, and add `"matchup"` to `_EdgeKind`:

```python
META_CAP = 0.10
COMFORT_CAP = 0.05
SYNERGY_CAP = 0.05
SIDE_CAP = 0.05
MATCHUP_CAP = 0.05
LOGISTIC_SLOPE = 1.2
STRONG_SIGNAL_GAMES = 30
SOME_SIGNAL_GAMES = 10

_EdgeKind = Literal["meta", "comfort", "synergy", "matchup"]
```

Add `champion_roles: Mapping[str, str | None]` to `Scorer.__init__`'s signature (after `our_side: Side,`) and store it:

```python
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
```

Add `_matchup_edge`, after `_synergy_edge`:

```python
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
```

Update `_candidate_edges` to accept and use the opposing side's picks:

```python
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
```

Update `_team_edges` the same way:

```python
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
```

Update `_strength_gap`'s two `_team_edges` calls:

```python
    def _strength_gap(self, our_picks: Sequence[str], opponent_picks: Sequence[str]) -> float:
        ours = sum(
            edge.edge for edge in self._team_edges(self._our_team_id, our_picks, opponent_picks)
        )
        theirs = sum(
            edge.edge for edge in self._team_edges(self._opponent_team_id, opponent_picks, our_picks)
        )
        return self._side_edge()[0] + ours - theirs
```

Update `score_pick`'s and `score_ban`'s `_candidate_edges` calls:

```python
        edges = self._candidate_edges(self._our_team_id, champion_id, our_picks, opponent_picks)
```

(in `score_pick`), and:

```python
        edges = self._candidate_edges(self._opponent_team_id, champion_id, opponent_picks, our_picks)
```

(in `score_ban`, replacing the existing line).

Update `composition_factors`'s two `_team_edges` calls:

```python
        factors.extend(
            self._factor(edge, "ours", _edge_impact(edge.edge))
            for edge in self._team_edges(self._our_team_id, our_picks, opponent_picks)
        )
        factors.extend(
            self._factor(edge, "theirs", -_edge_impact(edge.edge))
            for edge in self._team_edges(self._opponent_team_id, opponent_picks, our_picks)
        )
```

Add a `matchup` branch to `_label`, after the existing `synergy` branch (before `return record = ...`; the synergy branch currently ends with `return f"{quality} pair for {team_name}: {name} and {partner} ({record})"` — insert the new branch before that whole `if edge.kind == "synergy":` block or after it, at the same indentation level):

```python
        if edge.kind == "matchup":
            opponent_name = self._name(edge.partner_id or "")
            verb = "beats" if edge.edge >= 0 else "loses to"
            subject = {
                "candidate": f"Lane matchup {verb} their {opponent_name}",
                "ours": f"Your {name} {verb} their {opponent_name} in lane",
                "theirs": f"Their {name} {verb} your {opponent_name} in lane",
            }
            return f"{subject[owner]} (won {edge.wins} of {edge.games})"
```

Add a matchup branch to `score_ban`'s post-processing loop (the `for index, edge in enumerate(edges):` block that already handles `"comfort"` and `"synergy"`), as an additional `elif`:

```python
            elif edge.kind == "matchup":
                partner = self._name(edge.partner_id or "")
                verb = "beats" if edge.edge >= 0 else "loses to"
                label = f"Lane matchup {verb} your {partner} (won {edge.wins} of {edge.games})"
                factors[index] = factors[index].model_copy(update={"label": label})
```

- [ ] **Step 4: Fix the existing `scorer` fixture and call sites so the rest of the suite stays green**

In `backend/tests/engine/test_scoring.py`, add `champion_roles={}` to the existing `scorer` fixture's `Scorer(...)` call (after `our_side="blue",`):

```python
    return Scorer(
        stats,
        champion_names=NAMES,
        our_team_id="c9",
        our_team_name="Cloud9",
        opponent_team_id="opp",
        opponent_team_name="Rivals",
        our_side="blue",
        champion_roles={},
    )
```

In `backend/app/engine/service.py:156-164`, add `champion_roles=self._champion_roles,` to the `Scorer(...)` construction inside `recommend`:

```python
        scorer = Scorer(
            stats,
            champion_names=self._champion_names,
            our_team_id=self._our_team_id,
            our_team_name=our_team.name if our_team else "Your team",
            opponent_team_id=opponent.team_id,
            opponent_team_name=opponent.name,
            our_side=request.our_side,
            champion_roles=self._champion_roles,
        )
```

In `backend/app/model/train.py`, inside `_scorer_probabilities`, compute the champion roles once from `train_games` (reusing `FeatureSpace`'s existing role-fitting logic instead of duplicating it) and pass them to every `Scorer(...)` construction:

```python
def _scorer_probabilities(
    examples: list[_PrefixExample], train_games: list[GameDraft]
) -> list[float]:
    aggregates = compute_aggregates(train_games)
    if not aggregates:
        return [0.5] * len(examples)
    champion_names = {
        action.champion_id: action.champion_name for game in train_games for action in game.actions
    }
    champion_roles = FeatureSpace.fit(train_games).champion_roles
    scorers: dict[tuple[str, str, str], Scorer] = {}
    probabilities: list[float] = []
    for example in examples:
        game = example.game
        key = (game.patch, game.blue_team_id, game.red_team_id)
        scorer = scorers.get(key)
        if scorer is None:
            patch = game.patch if game.patch in aggregates else None
            stats = blend_patches(aggregates, patch=patch)
            scorer = Scorer(
                stats,
                champion_names=champion_names,
                our_team_id=game.blue_team_id,
                our_team_name=game.blue_team_id,
                opponent_team_id=game.red_team_id,
                opponent_team_name=game.red_team_id,
                our_side="blue",
                champion_roles=champion_roles,
            )
            scorers[key] = scorer
```

(the rest of the function body is unchanged).

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd backend && uv run pytest tests/engine/test_scoring.py tests/engine/test_service.py tests/model/test_training.py -v`
Expected: PASS (all tests, including the pre-existing ones in `test_service.py` and `test_training.py`, which exercise the call sites touched in Step 4).

- [ ] **Step 6: Commit**

```bash
git add backend/app/engine/scoring.py backend/app/engine/service.py backend/app/model/train.py backend/tests/engine/test_scoring.py
git commit -m "feat: surface counter-matchup factor in statistical scorer"
```

---

### Task 4: Counter-matchup scalar feature for the NN

**Files:**
- Modify: `backend/app/model/features.py`
- Test: `backend/tests/model/test_features.py`

**Interfaces:**
- Consumes: `GameDraft.lineups` (same as the existing role/trait fitting in `FeatureSpace.fit`).
- Produces: one new scalar per side appended to `FeatureSpace.encode`'s output, automatically reflected in `FeatureSpace.input_size` (no `net.py`/`predictor.py` changes needed, since both already size themselves from `input_size`).

- [ ] **Step 1: Write the failing test**

Add to `backend/tests/model/test_features.py`:

```python
def test_feature_space_encodes_matchup_win_rate_between_same_role_picks(make_game: Any) -> None:
    game = make_game(blue_picks=["c1"], red_picks=["c2"]).model_copy(
        update={
            "lineups": [
                PlayerLine(team_id="c9", role="top", champion_id="c1"),
                PlayerLine(team_id="opp", role="top", champion_id="c2"),
            ],
        }
    )
    space = FeatureSpace.fit([game])
    state = DraftState(patch="16.1", actions=["c1", "c2"])
    encoded = space.encode(state)

    per_side_width = 5 + 7 + len(space.tag_names) + 4 + 6 + 1
    blue_start = 4 * len(space.champion_ids) + 21 + len(space.patches) + 1
    blue_matchup_index = blue_start + per_side_width - 1
    red_matchup_index = blue_start + 2 * per_side_width - 1

    assert red_matchup_index == len(encoded) - 1
    assert encoded[blue_matchup_index] == 1.0  # c1 beat c2 in their one recorded top matchup
    assert encoded[red_matchup_index] == 0.0  # c2 lost that matchup

    lone_pick_encoded = space.encode(DraftState(patch="16.1", actions=["c1"]))
    assert lone_pick_encoded[blue_matchup_index] == 0.0  # no opposing top pick yet
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && uv run pytest tests/model/test_features.py::test_feature_space_encodes_matchup_win_rate_between_same_role_picks -v`
Expected: FAIL (assertion errors — the new scalar doesn't exist yet, so `blue_matchup_index`/`red_matchup_index` point at whatever the last meta-field values are instead).

- [ ] **Step 3: Fit and encode the matchup signal**

In `backend/app/model/features.py`, add a field to `FeatureSpace`, after `role_profiles`:

```python
    role_profiles: dict[str, dict[Role, RoleProfile]]
    matchup_win_rates: dict[str, float] = Field(default_factory=dict)
```

In `FeatureSpace.fit`, add `matchup_counts: dict[tuple[str, str, str], list[int]] = {}` next to the other counters (`role_counts`, `role_values`) before the `for game in train_games:` loop. Inside that loop, directly after the existing `for player in game.lineups: ...` block (the one building `role_counts`/`role_values`), add:

```python
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
                    backward = matchup_counts.setdefault((role, red_champion, blue_champion), [0, 0])
                    backward[0] += 1
                    backward[1] += not blue_won
```

After that `for game in train_games:` loop ends (same place `champion_roles`/`role_profiles` are finalized), add:

```python
        matchup_win_rates = {
            f"{role}|{champion_id}|{opponent_id}": wins / games
            for (role, champion_id, opponent_id), (games, wins) in matchup_counts.items()
        }
```

Add `matchup_win_rates=matchup_win_rates,` to the `cls(...)` construction returned by `fit`.

In `encode`, inside the `for side in ("blue", "red"):` loop, after the existing block that appends the `_META_FIELDS` averages (the `for selected in (current_picks, bans[side]): for field in _META_FIELDS: ...` block — the last thing currently appended per side), add:

```python
            opponent_side = "red" if side == "blue" else "blue"
            opponent_picks = picks[opponent_side]
            matchup_rates: list[float] = []
            for champion_id in current_picks:
                champion_role = self.champion_roles.get(champion_id)
                if champion_role is None:
                    continue
                opponent_id = next(
                    (cid for cid in opponent_picks if self.champion_roles.get(cid) == champion_role),
                    None,
                )
                if opponent_id is None:
                    continue
                rate = self.matchup_win_rates.get(f"{champion_role}|{champion_id}|{opponent_id}")
                if rate is not None:
                    matchup_rates.append(rate)
            features.append(sum(matchup_rates) / len(matchup_rates) if matchup_rates else 0.0)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && uv run pytest tests/model/test_features.py -v`
Expected: PASS (all tests in the file, including the pre-existing `input_size`/round-trip tests — `FeatureSpace.model_validate_json(space.model_dump_json())` must still equal `space.model_dump()`, confirming the new field serializes correctly).

- [ ] **Step 5: Commit**

```bash
git add backend/app/model/features.py backend/tests/model/test_features.py
git commit -m "feat: encode counter-matchup win rate as an NN feature"
```

---

### Task 5: Full verification and PR

**Files:** none (verification only).

- [ ] **Step 1: Run the full backend gate**

```bash
cd backend
uv run ruff check .
uv run ruff format --check .
uv run basedpyright
uv run pytest -q
```

Expected: all four commands exit 0 with no errors.

- [ ] **Step 2: Retrain and record the before/after metrics**

```bash
uv run python -m app.pipeline.build
uv run python -m app.model.train
cat .derived/model_metrics.json
```

Copy the resulting `accuracy`/`auc`/`log_loss` for `model` and note it against the Milestone 0 baseline (accuracy 0.553, AUC 0.515, log loss 0.692) — this is the ablation number for "counter-matchup only," to go in the PR description per the spec's testing section. `model_metrics.json` itself stays gitignored; only the numbers go in the PR text.

- [ ] **Step 3: Push and open the PR**

```bash
git push -u origin feat/<N>-counter-matchup-signal
gh pr create \
  --title "feat: add per-role counter-matchup signal" \
  --body "Adds a per-role champion-vs-champion win-rate signal to the statistical scorer's contributing factors and the NN's feature vector (Milestone 1, item 1 of docs/superpowers/specs/2026-11-21-recommendation-model-improvements-design.md).

Before/after (model, held-out validation set):
- Baseline (Milestone 0): accuracy 0.553, AUC 0.515, log loss 0.692
- With counter-matchup only: accuracy <fill in>, AUC <fill in>, log loss <fill in>

Closes #<N>"
```

- [ ] **Step 4: Verify no data files are staged**

```bash
git diff --cached --name-only
git log -1 --stat
```

Expected: only `backend/app/...` and `backend/tests/...` files touched by this plan — no `.derived/`, no GRID data paths.
