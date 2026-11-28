# Scorer-Probability Feature and Order-Aware Pooling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Improve `DraftNet`'s validation AUC in two ordered experiments: first, feed the existing explainable statistical scorer's blue-side win probability into the network as an extra feature (cheap, self-contained); then, replace the network's position-blind mean-pooling with an order-aware self-attention block over the true 20-action draft sequence (bigger architecture change). Retrain and record AUC after each.

**Architecture:** Task 1 adds a fold-safe `aggregates` field to `FeatureSpace`, fit the same way every other historical statistic already is (from `train_games`, never from `vocabulary_games`), and has `encode()` call the existing `app.engine.scoring.Scorer` internally to append one scalar to the end of `EncodedState.continuous`. Because `continuous_size` is already derived from `len(encode(DraftState()).continuous)`, this needs zero changes to `DraftNet`, `predictor.py`, `app/engine/ga.py`, or `app/engine/service.py`. Task 2 replaces `EncodedState`'s four sentinel-padded slot-group arrays (`blue_picks`/`blue_bans`/`red_picks`/`red_bans`) with one 20-length `draft_sequence` array holding each action's champion index at its true `DRAFT_ORDER` position, and replaces `DraftNet`'s 4-group mean pool with a learned position embedding plus one self-attention block, mean-pooled afterward. This is a real tensor-shape change and touches `net.py`, `train.py`, and `predictor.py`.

**Tech Stack:** Python 3.12, PyTorch (`torch.nn.MultiheadAttention`, `torch.nn.Embedding`), pydantic v2, pytest, ruff, basedpyright.

## Global Constraints

- Baseline to beat, recorded in `backend/.derived/model_metrics.json` as of 2026-11-23 (gitignored — commit messages are the durable record): **model AUC 0.588**, blue-rate baseline AUC 0.5, statistical-scorer AUC 0.513.
- Task 1's scorer-probability feature **must be fold-safe**: `FeatureSpace.fit()` computes it from its `train_games` parameter only (the per-fold `history_games` in `_out_of_fold_features`, or `fit_games` for the single production `feature_space`) — never from `vocabulary_games`. This mirrors every other historical statistic already on `FeatureSpace` (`team_win_rates`, `matchup_win_rates`, `patch_champion_stats`).
- Task 1 stays self-contained inside `backend/app/model/features.py`: no signature changes to `DraftNet`, `_stack_encoded`, `WinPredictor`, `app/engine/ga.py`, or `app/engine/service.py`. The new scalar rides inside `EncodedState.continuous`.
- Task 2 is a genuine architecture change (tensor shapes in `net.py`/`train.py`/`predictor.py` all change) — that is expected, not a bug to avoid. Per this project's established practice (see the shared-embeddings-and-prefix-weighting plan's "No runtime ablation flag" constraint), it ships as its own commit with git history as the ablation axis — no dual code paths.
- `WinPredictor.load(path) -> WinPredictor` and `WinPredictor.predict(states) -> list[float]` keep the same public signature across both tasks, so `app/engine/ga.py` and `app/engine/service.py` need no changes in either task.
- `embedding_dim` must stay evenly divisible by the self-attention block's `attention_heads` (default 2) — the existing default of 16 already satisfies this.
- Existing gates (`uv run ruff check .`, `uv run ruff format --check .`, `uv run basedpyright`, `uv run pytest -q`, all from `backend/`) stay green after every task.
- TDD mandatory: write the failing test before the implementation in every step.
- Retrain on the real local dataset (`uv run python -m app.model.train` from `backend/`, with the already-built `.derived/drafts.json`) after each task's gate passes, and record the resulting `model`/`statistical_scorer` AUC from `.derived/model_metrics.json` against the running baseline before moving to the next task.

---

### Task 1: Fold-safe statistical-scorer probability as a `FeatureSpace`-internal feature

**Files:**
- Modify: `backend/app/model/features.py`
- Test: `backend/tests/model/test_features.py`

**Interfaces:**
- Consumes: `app.engine.scoring.Scorer`, `app.engine.stats.{BlendedStats, blend_patches}`, `app.pipeline.aggregates.{PatchAggregate, compute_aggregates}` (all existing, unchanged).
- Produces: `FeatureSpace.aggregates: dict[str, PatchAggregate]` (new field, fit from `train_games`). `FeatureSpace.encode(state) -> EncodedState` unchanged in signature; `EncodedState.continuous` gains one trailing scalar (the blue-side scorer probability), so `FeatureSpace.continuous_size` increases by 1 automatically.

- [ ] **Step 1: Write the failing test for fold-safe aggregate fitting**

```python
# backend/tests/model/test_features.py — add near the other FeatureSpace.fit tests

def test_feature_space_fits_aggregates_from_training_games_only(make_game: Any) -> None:
    from app.pipeline.aggregates import compute_aggregates

    history_games = [make_game(blue_picks=["c0"], red_picks=["c1"]) for _ in range(3)]
    future_games = [
        make_game(patch="16.2", blue_picks=["c2"], red_picks=["c3"]) for _ in range(3)
    ]

    feature_space = FeatureSpace.fit(
        history_games, vocabulary_games=[*history_games, *future_games]
    )

    assert feature_space.aggregates == compute_aggregates(history_games)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && uv run pytest tests/model/test_features.py::test_feature_space_fits_aggregates_from_training_games_only -v`
Expected: FAIL with `AttributeError: 'FeatureSpace' object has no attribute 'aggregates'`.

- [ ] **Step 3: Add the `aggregates` field and fit it from `train_games`**

In `backend/app/model/features.py`, add to the imports near the top:

```python
from app.engine.scoring import Scorer
from app.engine.stats import BlendedStats, blend_patches
from app.pipeline.aggregates import PatchAggregate, compute_aggregates
```

Add a new field on `FeatureSpace`, immediately after the existing `team_win_rates` field:

```python
    team_win_rates: dict[str, float] = Field(default_factory=dict)
    aggregates: dict[str, PatchAggregate] = Field(default_factory=dict)
```

In `FeatureSpace.fit()`, add one line to the final `return cls(...)` call (the fitted-stats block at the end of the method):

```python
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
            aggregates=compute_aggregates(train_games),
        )
```

Note `train_games` (not `vocabulary_games`) is the same parameter every other historical stat above already uses — this is what makes the new field fold-safe for free.

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && uv run pytest tests/model/test_features.py::test_feature_space_fits_aggregates_from_training_games_only -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/app/model/features.py backend/tests/model/test_features.py
git commit -m "feat: fit FeatureSpace.aggregates from training games only"
```

- [ ] **Step 6: Write the failing tests for the scorer-probability continuous feature**

```python
# backend/tests/model/test_features.py — add near the other encode() tests

def test_encode_appends_a_scorer_probability_as_the_last_continuous_feature(
    make_game: Any,
) -> None:
    # "strong" wins almost every game it's picked in; "weak" wins almost none.
    strong_wins = [
        make_game(blue_picks=["strong"], red_picks=["weak"], winner="blue") for _ in range(20)
    ]
    weak_wins = [
        make_game(blue_picks=["weak"], red_picks=["strong"], winner="blue") for _ in range(2)
    ]
    feature_space = FeatureSpace.fit([*strong_wins, *weak_wins])

    blue_pick_index = next(
        index
        for index, slot in enumerate(DRAFT_ORDER)
        if slot.side == "blue" and slot.action == "pick"
    )
    red_pick_index = next(
        index
        for index, slot in enumerate(DRAFT_ORDER)
        if slot.side == "red" and slot.action == "pick"
    )
    actions = ["filler"] * len(DRAFT_ORDER)
    actions[blue_pick_index] = "strong"
    actions[red_pick_index] = "weak"
    state = DraftState(patch="16.1", actions=actions, blue_team_id="c9", red_team_id="opp")

    encoded = feature_space.encode(state)

    assert 0.0 <= encoded.continuous[-1] <= 1.0
    assert encoded.continuous[-1] > 0.5


def test_encode_scorer_probability_defaults_to_neutral_without_training_data() -> None:
    feature_space = FeatureSpace.fit([])

    encoded = feature_space.encode(DraftState())

    assert encoded.continuous[-1] == 0.5


def test_encode_caches_blended_stats_by_patch_not_by_team_pair(
    make_game: Any, monkeypatch: Any
) -> None:
    from app.model import features as features_module

    games = [make_game(patch="16.1", blue_picks=["c0"], red_picks=["c1"]) for _ in range(5)]
    feature_space = FeatureSpace.fit(games)

    call_count = 0
    original_blend_patches = features_module.blend_patches

    def counting_blend_patches(*args: Any, **kwargs: Any) -> Any:
        nonlocal call_count
        call_count += 1
        return original_blend_patches(*args, **kwargs)

    monkeypatch.setattr(features_module, "blend_patches", counting_blend_patches)

    for team_index in range(10):
        feature_space.encode(
            DraftState(
                patch="16.1",
                actions=["c0", "c1"],
                blue_team_id=f"team-{team_index}",
                red_team_id=f"opp-{team_index}",
            )
        )

    assert call_count <= 1
```

- [ ] **Step 7: Run the new tests to verify they fail**

Run: `cd backend && uv run pytest tests/model/test_features.py -k "scorer_probability" -v`
Expected: All FAIL — the first two with an `IndexError`/assertion mismatch because `continuous` doesn't have the new trailing scalar yet, the caching one with an `AttributeError` once `_scorer_probability` is referenced (write it to call a not-yet-existing method so the failure is explicit, or skip straight to Step 8 and treat a wrong `continuous[-1]` value as the pre-implementation failure signal).

- [ ] **Step 8: Implement the cached blended-stats lookup and the scorer-probability helper**

In `backend/app/model/features.py`, add a new cached property immediately after the existing `_champion_index` cached property:

```python
    @cached_property
    def _champion_index(self) -> dict[str, int]:
        """Champion id -> embedding index, built once per instance instead of per encode() call."""
        return {champion_id: index for index, champion_id in enumerate(self.champion_ids)}

    @cached_property
    def _blended_stats_cache(self) -> dict[str | None, BlendedStats]:
        """Per-patch blended stats, populated lazily across encode() calls (not per team pair)."""
        return {}
```

Add a new method near `_meta_for`:

```python
    def _scorer_probability(self, state: DraftState, picks: dict[str, list[str]]) -> float:
        """Blue-side win probability from the explainable statistical scorer, as a model feature."""
        if not self.aggregates:
            return _TEAM_PRIOR_RATE
        patch = state.patch if state.patch in self.aggregates else None
        stats = self._blended_stats_cache.get(patch)
        if stats is None:
            stats = blend_patches(self.aggregates, patch=patch)
            self._blended_stats_cache[patch] = stats
        scorer = Scorer(
            stats,
            champion_names={},
            our_team_id=state.blue_team_id or "",
            our_team_name="",
            opponent_team_id=state.red_team_id or "",
            opponent_team_name="",
            our_side="blue",
            champion_roles=self.champion_roles,
        )
        return scorer.game_win_probability(picks["blue"], picks["red"])
```

In `encode()`, append the new scalar right before the final `return`:

```python
            team_id = state.blue_team_id if side == "blue" else state.red_team_id
            continuous.append(
                self.team_win_rates.get(team_id, _TEAM_PRIOR_RATE) if team_id else _TEAM_PRIOR_RATE
            )

        continuous.append(self._scorer_probability(state, picks))

        return EncodedState(
            blue_picks=slots["blue_picks"],
            blue_bans=slots["blue_bans"],
            red_picks=slots["red_picks"],
            red_bans=slots["red_bans"],
            continuous=continuous,
        )
```

(The `team_id = ...` / `continuous.append(...)` two lines above are the existing last lines of the `for side in ("blue", "red"):` loop — only the new `continuous.append(self._scorer_probability(state, picks))` line and nothing else changes in this block.)

- [ ] **Step 9: Run the new tests to verify they pass**

Run: `cd backend && uv run pytest tests/model/test_features.py -k "scorer_probability" -v`
Expected: All PASS.

- [ ] **Step 10: Run the full backend gate**

Run: `cd backend && uv run ruff check . && uv run ruff format --check . && uv run basedpyright && uv run pytest -q`
Expected: All green. `continuous_size` increased by 1 everywhere it's derived, so no other test should need changes — if any test hardcodes an exact `continuous_size`/width number, fix its expected value now.

- [ ] **Step 11: Commit**

```bash
git add backend/app/model/features.py backend/tests/model/test_features.py
git commit -m "feat: feed the statistical scorer's win probability into FeatureSpace.encode"
```

- [ ] **Step 12: Retrain and record the AUC delta**

Run: `cd backend && uv run python -m app.model.train` (requires `STL_DATA_DIR`/`.derived/drafts.json` already built via `uv run python -m app.pipeline.build`).
Read `backend/.derived/model_metrics.json`'s `metrics.model.auc` (and `accuracy`/`log_loss`) and compare against the 0.588 baseline. Record the before/after numbers (this file is gitignored, so note them in the commit message of Task 2 or in conversation — there is no PR in this plan to carry them).

---

### Task 2: Order-aware self-attention pooling over the true draft sequence

**Files:**
- Modify: `backend/app/model/features.py`
- Modify: `backend/app/model/net.py`
- Modify: `backend/app/model/train.py`
- Modify: `backend/app/model/predictor.py`
- Test: `backend/tests/model/test_features.py`
- Test: `backend/tests/model/test_training.py`

**Interfaces:**
- Consumes: `DRAFT_ORDER` from `app.engine.draft_order` (existing import in `features.py`/`train.py`).
- Produces: `EncodedState.draft_sequence: list[int]` (replaces `blue_picks`/`blue_bans`/`red_picks`/`red_bans`) — a `len(DRAFT_ORDER)`-length array of champion vocabulary indices in true draft-order position, sentinel-padded (`== num_champions`) for not-yet-drafted slots. `DraftNet(num_champions, continuous_size, sequence_length, *, embedding_dim=16, hidden_size=32, dropout=0.5, attention_heads=2)` with `forward(draft_sequence: Tensor, continuous: Tensor) -> Tensor` (2 tensors instead of 5). Checkpoint gains a `sequence_length` field.

- [ ] **Step 1: Write the failing test for the new sequence shape**

```python
# backend/tests/model/test_features.py — replace test_encode_returns_sentinel_padded_slot_arrays_for_empty_state

def test_encode_returns_sentinel_padded_sequence_for_empty_state(
    synthetic_drafts: list[GameDraft],
) -> None:
    feature_space = FeatureSpace.fit(synthetic_drafts)
    sentinel = feature_space.num_champions

    encoded = feature_space.encode(DraftState())

    assert encoded.draft_sequence == [sentinel] * len(DRAFT_ORDER)
```

```python
# backend/tests/model/test_features.py — replace test_encode_places_champion_indices_in_the_matching_slot_group

def test_encode_places_champion_indices_at_their_draft_order_position(
    synthetic_drafts: list[GameDraft],
) -> None:
    feature_space = FeatureSpace.fit(synthetic_drafts)
    full_game = synthetic_drafts[0]
    action_ids = [action.champion_id for action in full_game.actions[: len(DRAFT_ORDER)]]
    state = DraftState(patch=full_game.patch, actions=action_ids)

    encoded = feature_space.encode(state)

    champion_index = {
        champion_id: index for index, champion_id in enumerate(feature_space.champion_ids)
    }
    expected = [
        champion_index.get(champion_id, feature_space.num_champions) for champion_id in action_ids
    ]
    expected += [feature_space.num_champions] * (len(DRAFT_ORDER) - len(expected))
    assert encoded.draft_sequence == expected
```

Also update the existing `test_feature_space_fits_only_training_games_and_encodes_blue_orientation` test's slot assertions (it currently reads `encoded.blue_picks`/`encoded.blue_bans`/`encoded.red_bans`):

```python
    # Check the draft sequence at its known positions (actions=["c3","c4","c3","c4","c3","c4","c1"];
    # DRAFT_ORDER[0..5] are bans B,R,B,R,B,R, DRAFT_ORDER[6] is the first blue pick)
    assert encoded.draft_sequence[0] == c3_index
    assert encoded.draft_sequence[1] == c4_index
    assert encoded.draft_sequence[6] == c1_index
```

(This replaces the four `assert c1_index in encoded.blue_picks` / etc. lines in that test — everything else in that test is unchanged.)

- [ ] **Step 2: Run the updated tests to verify they fail**

Run: `cd backend && uv run pytest tests/model/test_features.py -k "sequence or blue_orientation" -v`
Expected: FAIL with `AttributeError: 'EncodedState' object has no attribute 'draft_sequence'`.

- [ ] **Step 3: Replace `EncodedState`'s slot-group fields with `draft_sequence`**

In `backend/app/model/features.py`, replace the `EncodedState` dataclass:

```python
@dataclass(frozen=True)
class EncodedState:
    """Champion-index draft sequence plus continuous aggregate features for one draft state."""

    draft_sequence: list[int]
    continuous: list[float] = field(default_factory=list)
```

Replace the body of `encode()` (keep the method signature and the `_scorer_probability` call at the end from Task 1 unchanged):

```python
    def encode(self, state: DraftState) -> EncodedState:
        """Encode a draft state as a champion-index draft sequence plus continuous features."""
        champion_index = self._champion_index
        sentinel = self.num_champions
        sequence = [sentinel] * len(DRAFT_ORDER)
        picks: dict[str, list[str]] = {"blue": [], "red": []}
        bans: dict[str, list[str]] = {"blue": [], "red": []}

        for index, champion_id in enumerate(state.actions[: len(DRAFT_ORDER)]):
            slot = DRAFT_ORDER[index]
            sequence[index] = champion_index.get(champion_id, sentinel)
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

        continuous.append(self._scorer_probability(state, picks))

        return EncodedState(
            draft_sequence=sequence,
            continuous=continuous,
        )
```

- [ ] **Step 4: Run the updated tests to verify they pass**

Run: `cd backend && uv run pytest tests/model/test_features.py -v`
Expected: All pass except tests in other files that still reference the old `DraftNet`/tensor shapes (expected until Steps 5-8 below).

- [ ] **Step 5: Write the failing test for `DraftNet`'s new constructor and forward signature**

```python
# backend/tests/model/test_training.py — replace test_draft_net_returns_bounded_batch_probabilities

def test_draft_net_returns_bounded_batch_probabilities() -> None:
    model = DraftNet(
        num_champions=10,
        continuous_size=4,
        sequence_length=20,
        embedding_dim=4,
        hidden_size=8,
        dropout=0.1,
        attention_heads=2,
    )
    draft_sequence = torch.full((3, 20), 10, dtype=torch.long)  # sentinel == num_champions

    probabilities = model(draft_sequence, torch.zeros((3, 4)))

    assert probabilities.shape == (3,)
    assert torch.all(probabilities >= 0)
    assert torch.all(probabilities <= 1)
```

- [ ] **Step 6: Run test to verify it fails**

Run: `cd backend && uv run pytest tests/model/test_training.py::test_draft_net_returns_bounded_batch_probabilities -v`
Expected: FAIL with a `TypeError` about an unexpected keyword argument `sequence_length`.

- [ ] **Step 7: Replace `DraftNet` with the self-attention variant**

Replace the full contents of `backend/app/model/net.py`:

```python
"""PyTorch network used as the draft search fitness function."""

import torch
from torch import Tensor, nn


class DraftNet(nn.Module):
    """MLP over a self-attention-pooled draft sequence and continuous aggregate features."""

    def __init__(
        self,
        num_champions: int,
        continuous_size: int,
        sequence_length: int,
        *,
        embedding_dim: int = 16,
        hidden_size: int = 32,
        dropout: float = 0.5,
        attention_heads: int = 2,
    ) -> None:
        super().__init__()
        self.champion_embedding = nn.Embedding(num_champions + 1, embedding_dim)
        self.position_embedding = nn.Embedding(sequence_length, embedding_dim)
        self.attention = nn.MultiheadAttention(
            embedding_dim, attention_heads, dropout=dropout, batch_first=True
        )
        self.attention_norm = nn.LayerNorm(embedding_dim)
        input_size = embedding_dim + continuous_size
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid(),
        )

    def forward(self, draft_sequence: Tensor, continuous: Tensor) -> Tensor:
        """Return one blue-side win probability per input row."""
        positions = torch.arange(draft_sequence.shape[1], device=draft_sequence.device)
        embedded = self.champion_embedding(draft_sequence) + self.position_embedding(positions)
        attended, _ = self.attention(embedded, embedded, embedded)
        sequence_repr = self.attention_norm(embedded + attended)
        pooled = sequence_repr.mean(dim=1)
        combined = torch.cat([pooled, continuous], dim=1)
        return self.layers(combined).squeeze(-1)
```

- [ ] **Step 8: Run test to verify it passes**

Run: `cd backend && uv run pytest tests/model/test_training.py::test_draft_net_returns_bounded_batch_probabilities -v`
Expected: PASS.

- [ ] **Step 9: Run the full backend gate and note the expected fallout**

Run: `cd backend && uv run ruff check . && uv run ruff format --check . && uv run basedpyright && uv run pytest -q`
Expected: `ruff`/`format`/`basedpyright` clean. `pytest` will show failures in `tests/model/test_training.py` for every test that trains a real model (they call `train_model`, which still builds the old 5-tensor shape) — leave those for Step 10.

- [ ] **Step 10: Wire `train.py` to the 2-tensor shape**

In `backend/app/model/train.py`, replace `_stack_encoded`:

```python
def _stack_encoded(
    torch: Any, encoded_states: Sequence[EncodedState], device: Any
) -> tuple[Any, Any]:
    """Stack a batch of EncodedState into the two tensors DraftNet.forward expects."""
    return (
        torch.tensor(
            [state.draft_sequence for state in encoded_states], dtype=torch.long, device=device
        ),
        torch.tensor(
            [state.continuous for state in encoded_states], dtype=torch.float32, device=device
        ),
    )
```

Replace the model construction and the training-tensor unpacking in `train_model`:

```python
    model = DraftNet(
        feature_space.num_champions,
        feature_space.continuous_size,
        len(DRAFT_ORDER),
        embedding_dim=embedding_dim,
        hidden_size=hidden_size,
        dropout=dropout,
    ).to(target_device)
    draft_sequence_tensor, continuous_tensor = _stack_encoded(torch, train_encoded, target_device)
```

Replace the per-batch model call inside the mini-batch loop:

```python
            probabilities = model(
                draft_sequence_tensor[batch_indices],
                continuous_tensor[batch_indices],
            )
```

(`stopping_tensors = _stack_encoded(...)` / `model(*stopping_tensors)` and `validation_tensors = _stack_encoded(...)` / `model(*validation_tensors)` need no changes — they already unpack whatever `_stack_encoded` returns.)

Add `sequence_length` to the checkpoint save block:

```python
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.cpu().state_dict(),
            "num_champions": feature_space.num_champions,
            "continuous_size": feature_space.continuous_size,
            "embedding_dim": embedding_dim,
            "sequence_length": len(DRAFT_ORDER),
            "hidden_size": hidden_size,
            "dropout": dropout,
            "feature_space": feature_space.model_dump(mode="json"),
        },
        output_dir / "model.pt",
    )
```

- [ ] **Step 11: Wire `predictor.py` to the 2-tensor shape**

In `backend/app/model/predictor.py`, update `WinPredictor.load`:

```python
        try:
            feature_space = FeatureSpace.model_validate(checkpoint["feature_space"])
            num_champions = int(checkpoint["num_champions"])
            continuous_size = int(checkpoint["continuous_size"])
            embedding_dim = int(checkpoint["embedding_dim"])
            sequence_length = int(checkpoint["sequence_length"])
            hidden_size = int(checkpoint["hidden_size"])
            dropout = float(checkpoint["dropout"])
            state_dict = checkpoint["state_dict"]
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"Invalid model checkpoint: {path}") from error
        if num_champions != feature_space.num_champions:
            raise ValueError("Model champion count does not match the saved feature space")
        if continuous_size != feature_space.continuous_size:
            raise ValueError("Model continuous feature size does not match the saved feature space")

        model = DraftNet(
            num_champions,
            continuous_size,
            sequence_length,
            embedding_dim=embedding_dim,
            hidden_size=hidden_size,
            dropout=dropout,
        )
```

Update `WinPredictor.predict`:

```python
    def predict(self, states: Sequence[DraftState]) -> list[float]:
        """Predict blue-side game win probabilities for a batch of draft states.

        Uses GPU (CUDA) when available for faster inference.
        """
        if not states:
            return []
        torch = _torch_module()
        encoded = [self.feature_space.encode(state) for state in states]
        draft_sequence = torch.tensor(
            [state.draft_sequence for state in encoded], dtype=torch.long, device=self._device
        )
        continuous = torch.tensor(
            [state.continuous for state in encoded], dtype=torch.float32, device=self._device
        )
        with torch.inference_mode():
            predictions = self._model(draft_sequence, continuous)
        if self._device != "cpu":
            predictions = predictions.cpu()
        return [float(probability) for probability in predictions.tolist()]
```

- [ ] **Step 12: Update the checkpoint-fields test**

```python
# backend/tests/model/test_training.py — update test_train_model_saves_embedding_checkpoint_fields

def test_train_model_saves_embedding_checkpoint_fields(make_game: Any, tmp_path: Path) -> None:
    games = [_planted_game(make_game, index, index % 2 == 0) for index in range(24)]

    train_model(games, tmp_path, epochs=1, seed=17, device="cpu")

    checkpoint = torch.load(tmp_path / "model.pt", weights_only=True)
    assert checkpoint["num_champions"] > 0
    assert checkpoint["continuous_size"] > 0
    assert checkpoint["embedding_dim"] == 16
    assert checkpoint["sequence_length"] == len(DRAFT_ORDER)
    assert "input_size" not in checkpoint
```

- [ ] **Step 13: Run the full backend gate**

Run: `cd backend && uv run ruff check . && uv run ruff format --check . && uv run basedpyright && uv run pytest -q`
Expected: All green. If any other test in `tests/model/test_training.py` still constructs `DraftNet` with the old keyword arguments or feeds the model a single tensor, update it the same way as Steps 5-8 before considering this task done — do not leave any test red.

- [ ] **Step 14: Commit**

```bash
git add backend/app/model/features.py backend/app/model/net.py backend/app/model/train.py \
  backend/app/model/predictor.py backend/tests/model/test_features.py backend/tests/model/test_training.py
git commit -m "feat: replace mean-pooled champion groups with order-aware self-attention over the draft sequence"
```

- [ ] **Step 15: Retrain and record the final AUC comparison**

Run: `cd backend && uv run python -m app.model.train`.
Read `backend/.derived/model_metrics.json`'s `metrics.model.auc`/`accuracy`/`log_loss` and compare against both the pre-Task-1 baseline (0.588 AUC) and the Task 1 result recorded in that task's Step 12. State plainly whether the order-aware pooling helped, hurt, or was a wash relative to Task 1 — the goal of this plan is measuring the delta, not guaranteeing an improvement.

## Self-Review

- **Spec coverage:** Task 1 adds the scorer probability as a fold-safe `FeatureSpace`-internal feature with zero `DraftNet`/`predictor.py` changes ✓; Task 2 replaces mean-pooling with position-embedding + self-attention over the true draft order ✓; both retrain-and-record steps are present (Task 1 Step 12, Task 2 Step 15) so the user's requested ordering — ship the scorer feature first, measure, then try the pooling change, measure again — is preserved ✓.
- **Fold safety:** `FeatureSpace.aggregates` is fit from the same `train_games` parameter as `team_win_rates`/`matchup_win_rates`, so `_out_of_fold_features`'s existing 5-fold refit loop makes the scorer-probability feature out-of-fold automatically, with no new fold-handling code needed. ✓
- **Blast radius:** confirmed by grep that the only call sites touching `EncodedState`'s slot fields or `DraftNet`'s constructor/forward are `features.py`, `net.py`, `train.py`, `predictor.py`, and their direct tests — `app/engine/ga.py`, `app/engine/service.py`, `app/model/tune.py`, and `app/model/diagnose.py` all go through `train_model`'s/`WinPredictor`'s stable public signatures and need no changes. ✓
- **Type/name consistency:** `EncodedState.draft_sequence`, `DraftNet(num_champions, continuous_size, sequence_length, ...)`, and `_stack_encoded`'s 2-tuple return are used identically across `features.py`, `net.py`, `train.py`, `predictor.py`, and both test files. ✓
- **Known limitation to carry into any future write-up:** Task 1's scorer probability is itself derived partly from team-comfort and synergy history, which already overlaps with Task 2's self-attention signal once champion identity and order are learnable — if Task 2 doesn't improve AUC over Task 1 alone, that overlap (not a broken implementation) is the most likely explanation.
