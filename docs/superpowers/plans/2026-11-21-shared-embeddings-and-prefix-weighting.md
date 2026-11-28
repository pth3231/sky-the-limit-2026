# Shared Champion Embeddings and Prefix Down-Weighting Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the one-hot champion presence encoding with shared learned champion embeddings, and down-weight near-empty draft prefixes during training, so the win-probability network has a chance to generalize instead of memorizing training games.

**Architecture:** `FeatureSpace.encode()` stops emitting a one-hot block over the full champion vocabulary and instead emits four fixed-length arrays of champion vocabulary indices (blue picks, blue bans, red picks, red bans), alongside the existing continuous aggregate features (role/meta/trait stats), unchanged. `DraftNet` gains an `nn.Embedding` table shared across all four slot groups, mean-pools each group to one vector, concatenates the four pooled vectors with the continuous features, then feeds the existing MLP body. `train.py` and `predictor.py` are updated to build and consume this two-part shape. Separately, `_prefix_examples` gains a per-example weight proportional to how many draft actions are locked in, applied as a sample weight in the BCE loss (training and stopping) so near-empty prefixes stop dominating the loss.

**Tech Stack:** Python 3.12, PyTorch (`torch.nn.Embedding`, `torch.nn.functional.binary_cross_entropy` with `reduction="none"`), pydantic v2, pytest, ruff, basedpyright.

## Global Constraints

- Baseline to compare against: **AUC 0.515** (recorded pre-Milestone-1 in `backend/.derived/model_metrics.json`, itself gitignored — PR descriptions are the durable record).
- `DRAFT_ORDER` (from `app/engine/draft_order.py`) has exactly **5 blue bans, 5 red bans, 5 blue picks, 5 red picks** (20 slots total; verified from `_PHASES = (("ban", "BRBRBR"), ("pick", "BRRBBR"), ("ban", "RBRB"), ("pick", "RBBR"))`). Each of the four embedding slot groups (blue picks, blue bans, red picks, red bans) is therefore a **fixed-length array of 5 integer indices**, padded with a sentinel index when a slot isn't filled yet in a partial draft state.
- Embedding sentinel/padding index is **`len(champion_ids)`** — one past the last valid champion vocabulary index (champion IDs are 0-indexed `0..len(champion_ids)-1`), so the embedding table size is `len(champion_ids) + 1`.
- `embedding_dim` default: **16**. Justification: the champion vocabulary is roughly 150-200 entries; at `embedding_dim=16` the embedding table has on the order of `(200 + 1) * 16 ≈ 3,200` parameters, far smaller than the one-hot path's `4 * 200 = 800`-wide input feeding a 32-unit hidden layer (`800 * 32 = 25,600` first-layer weights just for that block). A low-data regime (~1,273 independent games) calls for a small embedding width — far smaller than the 50-300 typically used in NLP contexts.
- Prefix down-weighting formula: **`weight = prefix_length / len(DRAFT_ORDER)`** (linear ramp from `0.0` at an empty draft to `1.0` at a complete 20-action draft). Applied as a per-example weight in a **weighted mean** BCE loss: `(per_example_loss * weights).sum() / weights.sum().clamp_min(1e-8)` — never `reduction="mean"` with a `weight=` tensor, since that divides by example count instead of weight sum and silently shrinks the loss scale.
- **No runtime ablation flag.** Each change (embeddings, prefix weighting) ships as its own task/commit/PR, same pattern already used for the counter-matchup feature in this project's history. The ablation required by the spec's exit criteria is measured by retraining before and after each PR and recording the AUC delta in that PR's description — not by a `use_embeddings`/`use_prefix_weighting` toggle. Maintaining two parallel `encode()`/`DraftNet`/training-loop code paths side by side would double the surface this plan has to build and test for a change with a one-off evaluation purpose; git history already gives a clean, zero-code-cost ablation axis.
- Existing gates (`uv run ruff check .`, `uv run ruff format --check .`, `uv run basedpyright`, `uv run pytest -q`, all from `backend/`) must stay green after every task.
- TDD mandatory: write the failing test before the implementation in every step.
- One GitHub issue per task, created at execution time per this repo's issue-first convention (not part of this plan document).

---

### Task 1: Replace one-hot presence with champion-index slot arrays in `FeatureSpace`

**Files:**
- Modify: `backend/app/model/features.py`
- Test: `backend/tests/model/test_features.py`

**Interfaces:**
- Consumes: `DraftState(patch: str, actions: list[str])` (unchanged), `DRAFT_ORDER` from `app.engine.draft_order` (unchanged), `FeatureSpace.champion_ids: list[str]` (unchanged, already fitted vocabulary).
- Produces: `EncodedState` dataclass with fields `blue_picks: list[int]`, `blue_bans: list[int]`, `red_picks: list[int]`, `red_bans: list[int]` (each length 5, sentinel-padded), `continuous: list[float]` (everything `encode()` used to emit *except* the one-hot presence block — action-count one-hot, patch one-hot, and all per-side aggregate stats, unchanged in content and order). `FeatureSpace.encode(state: DraftState) -> EncodedState` (return type changes from `list[float]`). `FeatureSpace.num_champions` property (`int`, `= len(self.champion_ids)`). `FeatureSpace.continuous_size` property (`int`, `= len(self.encode(DraftState()).continuous)`). The old `FeatureSpace.input_size` property is removed — Task 3 updates its only two callers.

- [ ] **Step 1: Write the failing test for slot-array shape and padding**

```python
# backend/tests/model/test_features.py — add near the other FeatureSpace.encode tests

def test_encode_returns_sentinel_padded_slot_arrays_for_empty_state(
    synthetic_drafts: list[GameDraft],
) -> None:
    feature_space = FeatureSpace.fit(synthetic_drafts)
    sentinel = feature_space.num_champions

    encoded = feature_space.encode(DraftState())

    assert encoded.blue_picks == [sentinel] * 5
    assert encoded.blue_bans == [sentinel] * 5
    assert encoded.red_picks == [sentinel] * 5
    assert encoded.red_bans == [sentinel] * 5
```

`synthetic_drafts` is the existing fixture in `backend/tests/conftest.py` (already used by other `FeatureSpace`/`Scorer` tests in this suite) — import `FeatureSpace`, `DraftState` from `app.model.features` and `GameDraft` from `app.pipeline.dataset` at the top of the test file if not already imported.

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && uv run pytest tests/model/test_features.py::test_encode_returns_sentinel_padded_slot_arrays_for_empty_state -v`
Expected: FAIL with `AttributeError: 'FeatureSpace' object has no attribute 'num_champions'` (or a `list[float]` has no attribute `blue_picks` once `num_champions` is stubbed) — confirms the new shape doesn't exist yet.

- [ ] **Step 3: Write the failing test for a fully populated draft**

```python
def test_encode_places_champion_indices_in_the_matching_slot_group(
    synthetic_drafts: list[GameDraft],
) -> None:
    feature_space = FeatureSpace.fit(synthetic_drafts)
    full_game = synthetic_drafts[0]
    state = DraftState(
        patch=full_game.patch,
        actions=[action.champion_id for action in full_game.actions],
    )

    encoded = feature_space.encode(state)

    champion_index = {
        champion_id: index for index, champion_id in enumerate(feature_space.champion_ids)
    }
    expected_blue_picks = sorted(
        champion_index[action.champion_id]
        for action in full_game.actions
        if action.team_id == full_game.blue_team_id and action.action_type == "pick"
    )
    assert sorted(value for value in encoded.blue_picks if value != feature_space.num_champions) == (
        expected_blue_picks
    )
    assert len(encoded.blue_picks) == 5
```

`synthetic_drafts`'s `make_game` fixture (see `backend/tests/conftest.py`) builds games via blue/red pick/ban lists, so `synthetic_drafts[0]` always has a full set of picks and bans for this assertion to be meaningful.

- [ ] **Step 4: Run both new tests to verify they fail**

Run: `cd backend && uv run pytest tests/model/test_features.py -k "slot_arrays or matching_slot_group" -v`
Expected: Both FAIL (the second with the same `AttributeError`/shape mismatch as Step 2).

- [ ] **Step 5: Implement `EncodedState` and the new `encode()`**

In `backend/app/model/features.py`, add `EncodedState` right after the `DraftState` class:

```python
from dataclasses import dataclass, field


@dataclass(frozen=True)
class EncodedState:
    """Champion-index slot arrays plus continuous aggregate features for one draft state."""

    blue_picks: list[int]
    blue_bans: list[int]
    red_picks: list[int]
    red_bans: list[int]
    continuous: list[float] = field(default_factory=list)
```

Replace the `input_size` property and the body of `encode()`:

```python
    @property
    def num_champions(self) -> int:
        """Return the fitted champion vocabulary size (the embedding padding index)."""
        return len(self.champion_ids)

    @property
    def continuous_size(self) -> int:
        """Return the stable width of the continuous feature block for this feature space."""
        return len(self.encode(DraftState()).continuous)

    def _meta_for(self, patch: str, champion_id: str) -> ChampionMeta | None:
        return self.patch_champion_stats.get(patch, {}).get(
            champion_id,
            self.global_champion_stats.get(champion_id),
        )

    def encode(self, state: DraftState) -> EncodedState:
        """Encode a draft state as champion-index slot arrays plus continuous features."""
        champion_index = {champion_id: index for index, champion_id in enumerate(self.champion_ids)}
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

        return EncodedState(
            blue_picks=slots["blue_picks"],
            blue_bans=slots["blue_bans"],
            red_picks=slots["red_picks"],
            red_bans=slots["red_bans"],
            continuous=continuous,
        )
```

This removes the `presence` one-hot block entirely and the `champion_count`/`presence` local variables; everything else (`action_count`, `patch_features`, and the per-side aggregate loop) is unchanged in logic, just renamed from `features` to `continuous` and returned inside `EncodedState` instead of a flat list. Note the slot-filling loop now tracks a per-group `cursor` so that if a side bans or picks fewer than 5 champions (shouldn't happen in a complete `DRAFT_ORDER`-length state, but matters for partial states), the remaining entries stay at the sentinel.

- [ ] **Step 6: Run the new tests to verify they pass**

Run: `cd backend && uv run pytest tests/model/test_features.py -k "slot_arrays or matching_slot_group" -v`
Expected: Both PASS.

- [ ] **Step 7: Run the full `test_features.py` file to check for regressions**

Run: `cd backend && uv run pytest tests/model/test_features.py -v`
Expected: All pass except any test that directly asserted on the old flat `list[float]` return shape of `encode()` or on `input_size` — fix those now by updating their assertions to use `EncodedState`'s fields / `continuous_size` (there should be at most one or two such tests; read the failures and adjust the assertions to match the new return shape, keeping the same intent — e.g. a test that checked `len(feature_space.encode(state)) == some_value` becomes `len(feature_space.encode(state).continuous) == some_value - (4 * feature_space.num_champions)` only if it was asserting the old combined width; if it was asserting something about the aggregate block specifically, assert on `.continuous` directly instead).

- [ ] **Step 8: Run the full backend gate**

Run: `cd backend && uv run ruff check . && uv run ruff format --check . && uv run basedpyright && uv run pytest -q`
Expected: `ruff`/`format`/`basedpyright` clean. `pytest` will show failures in `backend/app/model/train.py` and `backend/app/model/predictor.py` callers of `.encode()`/`.input_size` — those are fixed in Task 3, not this task. Confirm the only failures are in `tests/model/test_training.py` and nothing in `tests/model/test_features.py` or other test files; if anything else fails, fix it before moving on.

- [ ] **Step 9: Commit**

```bash
git add backend/app/model/features.py backend/tests/model/test_features.py
git commit --trailer "Closes: #<ISSUE_NUMBER>" -m "feat: encode champion slots as index arrays instead of one-hot"
```

---

### Task 2: Give `DraftNet` a shared champion embedding table

**Files:**
- Modify: `backend/app/model/net.py`
- Test: `backend/tests/model/test_training.py` (the existing `test_draft_net_returns_bounded_batch_probabilities` test lives here and must be updated to the new constructor/forward signature as part of this task)

**Interfaces:**
- Consumes: nothing from Task 1 directly (this task only changes `DraftNet`'s shape; it is wired to `EncodedState` in Task 3).
- Produces: `DraftNet(num_champions: int, continuous_size: int, *, embedding_dim: int = 16, hidden_size: int = 32, dropout: float = 0.5)`. `DraftNet.forward(self, blue_picks: Tensor, blue_bans: Tensor, red_picks: Tensor, red_bans: Tensor, continuous: Tensor) -> Tensor` where `blue_picks`/`blue_bans`/`red_picks`/`red_bans` are `(batch, 5)` `long` tensors of champion indices (padding sentinel `== num_champions`) and `continuous` is a `(batch, continuous_size)` `float32` tensor; returns `(batch,)` win probabilities.

- [ ] **Step 1: Write the failing test for the new constructor and forward signature**

```python
# backend/tests/model/test_training.py — replace the existing test_draft_net_returns_bounded_batch_probabilities

def test_draft_net_returns_bounded_batch_probabilities() -> None:
    model = DraftNet(num_champions=10, continuous_size=4, embedding_dim=4, hidden_size=8, dropout=0.1)
    empty_slot = torch.full((3, 5), 10, dtype=torch.long)  # sentinel == num_champions

    probabilities = model(
        empty_slot,
        empty_slot,
        empty_slot,
        empty_slot,
        torch.zeros((3, 4)),
    )

    assert probabilities.shape == (3,)
    assert torch.all(probabilities >= 0)
    assert torch.all(probabilities <= 1)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && uv run pytest tests/model/test_training.py::test_draft_net_returns_bounded_batch_probabilities -v`
Expected: FAIL with a `TypeError` about unexpected keyword argument `num_champions` (the current constructor only takes `input_size`).

- [ ] **Step 3: Implement the embedding-based `DraftNet`**

Replace the full contents of `backend/app/model/net.py`:

```python
"""PyTorch network used as the draft search fitness function."""

import torch
from torch import Tensor, nn


class DraftNet(nn.Module):
    """MLP over mean-pooled champion embeddings and continuous aggregate features."""

    def __init__(
        self,
        num_champions: int,
        continuous_size: int,
        *,
        embedding_dim: int = 16,
        hidden_size: int = 32,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.champion_embedding = nn.Embedding(num_champions + 1, embedding_dim)
        input_size = 4 * embedding_dim + continuous_size
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

    def forward(
        self,
        blue_picks: Tensor,
        blue_bans: Tensor,
        red_picks: Tensor,
        red_bans: Tensor,
        continuous: Tensor,
    ) -> Tensor:
        """Return one blue-side win probability per input row."""
        pooled = [
            self.champion_embedding(group).mean(dim=1)
            for group in (blue_picks, blue_bans, red_picks, red_bans)
        ]
        combined = torch.cat([*pooled, continuous], dim=1)
        return self.layers(combined).squeeze(-1)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && uv run pytest tests/model/test_training.py::test_draft_net_returns_bounded_batch_probabilities -v`
Expected: PASS.

- [ ] **Step 5: Run the full backend gate**

Run: `cd backend && uv run ruff check . && uv run ruff format --check . && uv run basedpyright && uv run pytest -q`
Expected: `ruff`/`format`/`basedpyright` clean (note `net.py` now imports `torch` directly at module level instead of only `Tensor, nn` — this is fine since `net.py` is only ever imported lazily from inside `train.py`/`predictor.py` after `_torch_module()` has already confirmed torch is installed). `pytest` still shows the same Task-3-pending failures in `tests/model/test_training.py` for any test that constructs `DraftNet` with the old `input_size=` keyword or calls `model(features_tensor)` with one positional tensor — leave those for Task 3.

- [ ] **Step 6: Commit**

```bash
git add backend/app/model/net.py backend/tests/model/test_training.py
git commit --trailer "Closes: #<ISSUE_NUMBER>" -m "feat: pool shared champion embeddings in DraftNet instead of one-hot input"
```

---

### Task 3: Wire `train.py` and `predictor.py` to the two-part input shape

**Files:**
- Modify: `backend/app/model/train.py`
- Modify: `backend/app/model/predictor.py`
- Test: `backend/tests/model/test_training.py`

**Interfaces:**
- Consumes: `FeatureSpace.encode(state) -> EncodedState` and `FeatureSpace.num_champions`/`continuous_size` properties (Task 1); `DraftNet(num_champions, continuous_size, *, embedding_dim=16, hidden_size=32, dropout=0.5)` and its 5-tensor `forward` (Task 2).
- Produces: `model.pt` checkpoint dict with keys `state_dict`, `num_champions`, `continuous_size`, `embedding_dim`, `hidden_size`, `dropout`, `feature_space` (drops `input_size`). `WinPredictor.load`/`WinPredictor.predict` continue to expose the same public signatures (`load(path) -> WinPredictor`, `predict(states: Sequence[DraftState]) -> list[float]`) — only their internals change, so `app/engine/ga.py` and `app/engine/service.py` need no changes.

- [ ] **Step 1: Write the failing end-to-end test**

The existing `test_training_learns_planted_winning_champion_and_loads_batch_predictor` in `backend/tests/model/test_training.py` already exercises the full `train_model` -> `WinPredictor.load` -> `predict` path end to end with no assumptions about the internal tensor shape — it will simply fail against the current (pre-Task-3) code because `train_model` still builds one flat tensor. Treat that existing test as this task's primary regression test; no new test content is needed for the "does it still work" check. Add one new, narrower test that pins the checkpoint's new fields:

```python
def test_train_model_saves_embedding_checkpoint_fields(make_game: Any, tmp_path: Path) -> None:
    games = [_planted_game(make_game, index, index % 2 == 0) for index in range(24)]

    train_model(games, tmp_path, epochs=1, seed=17, device="cpu")

    checkpoint = torch.load(tmp_path / "model.pt", weights_only=True)
    assert checkpoint["num_champions"] > 0
    assert checkpoint["continuous_size"] > 0
    assert checkpoint["embedding_dim"] == 16
    assert "input_size" not in checkpoint
```

- [ ] **Step 2: Run both tests to verify they fail**

Run: `cd backend && uv run pytest tests/model/test_training.py::test_training_learns_planted_winning_champion_and_loads_batch_predictor tests/model/test_training.py::test_train_model_saves_embedding_checkpoint_fields -v`
Expected: Both FAIL — the first with an `AttributeError`/`TypeError` from `train_model` still calling the old `feature_space.encode(...)` as a flat vector or `DraftNet(input_size=...)`; the second the same way.

- [ ] **Step 3: Update `_out_of_fold_features` and the encoding call sites in `train.py`**

In `backend/app/model/train.py`, change the import and the return type/body of `_out_of_fold_features`:

```python
from app.model.features import DraftState, EncodedState, FeatureSpace
```

```python
def _out_of_fold_features(
    examples: list[_PrefixExample],
    games: list[GameDraft],
    ddragon: dict[str, Any] | None = None,
) -> list[EncodedState]:
    """Encode each example with historical stats fitted outside its series fold."""
    series_games: dict[str, list[GameDraft]] = defaultdict(list)
    for game in games:
        series_games[game.series_id].append(game)
    ordered_series = sorted(
        series_games, key=lambda series_id: _series_key(series_games[series_id])
    )
    if not ordered_series:
        return []

    fold_count = min(5, len(ordered_series))
    series_folds = {series_id: index % fold_count for index, series_id in enumerate(ordered_series)}
    examples_by_fold: dict[int, list[tuple[int, _PrefixExample]]] = defaultdict(list)
    for index, example in enumerate(examples):
        fold = series_folds.get(example.game.series_id)
        if fold is None:
            raise ValueError(
                f"Example series {example.game.series_id!r} is absent from training games"
            )
        examples_by_fold[fold].append((index, example))

    encoded: list[EncodedState | None] = [None] * len(examples)
    for fold, fold_examples in examples_by_fold.items():
        history_games = [game for game in games if series_folds[game.series_id] != fold]
        feature_space = FeatureSpace.fit(
            history_games,
            ddragon,
            vocabulary_games=games,
        )
        for index, example in fold_examples:
            encoded[index] = feature_space.encode(example.state)

    if any(value is None for value in encoded):
        raise RuntimeError("Not all training examples received out-of-fold features")
    return [value for value in encoded if value is not None]
```

Note: this keeps using a *separately refit* `FeatureSpace` per fold (as before) purely to compute each example's `EncodedState`; the champion vocabulary (and therefore the embedding sentinel index) must still come from the single `feature_space` fitted once in `train_model` on `fit_games` (assigned a few lines later) for the actual `num_champions`/`continuous_size` passed to `DraftNet` — the per-fold refits here only affect the continuous aggregate stats and champion index numbering must stay consistent. Since `FeatureSpace.fit(..., vocabulary_games=games)` always builds `champion_ids` from the same `vocabulary_games` argument regardless of fold, the index numbering is already consistent across folds — no change needed beyond the return type.

- [ ] **Step 4: Add a tensor-stacking helper and rewrite the training loop's tensor construction**

Add this helper directly above `train_model` in `backend/app/model/train.py`:

```python
def _stack_encoded(
    torch: Any, encoded_states: Sequence[EncodedState], device: Any
) -> tuple[Any, Any, Any, Any, Any]:
    """Stack a batch of EncodedState into the five tensors DraftNet.forward expects."""
    return (
        torch.tensor([state.blue_picks for state in encoded_states], dtype=torch.long, device=device),
        torch.tensor([state.blue_bans for state in encoded_states], dtype=torch.long, device=device),
        torch.tensor([state.red_picks for state in encoded_states], dtype=torch.long, device=device),
        torch.tensor([state.red_bans for state in encoded_states], dtype=torch.long, device=device),
        torch.tensor(
            [state.continuous for state in encoded_states], dtype=torch.float32, device=device
        ),
    )
```

Add a new import line `from collections.abc import Sequence` right after the existing `from collections import defaultdict, deque` line at the top of `backend/app/model/train.py` (confirmed: `train.py` does not currently import `Sequence` from anywhere).

In `train_model`, replace the block that builds `feature_space`/`input_size`/`train_features`/`stopping_features`/`validation_features` and the tensors built from them:

```python
    feature_space = FeatureSpace.fit(fit_games, ddragon)
    train_encoded = _out_of_fold_features(train_examples, fit_games, ddragon)
    train_labels = [example.label for example in train_examples]
    stopping_encoded = [feature_space.encode(example.state) for example in stopping_examples]
    stopping_labels = [example.label for example in stopping_examples]
    validation_encoded = [feature_space.encode(example.state) for example in validation_examples]
    validation_labels = [example.label for example in validation_examples]
```

(`train_examples`/`stopping_examples`/`validation_examples` keep the same names and come from the same `_prefix_examples` calls already present a few lines above this block — only the variable names `train_features` -> `train_encoded`, etc., and their construction change.)

Replace the model construction and the per-epoch tensor usage:

```python
    model = DraftNet(
        feature_space.num_champions,
        feature_space.continuous_size,
        hidden_size=hidden_size,
        dropout=dropout,
    ).to(target_device)
    blue_picks_tensor, blue_bans_tensor, red_picks_tensor, red_bans_tensor, continuous_tensor = (
        _stack_encoded(torch, train_encoded, target_device)
    )
    labels_tensor = torch.tensor(train_labels, dtype=torch.float32, device=target_device)
    stopping_tensors = _stack_encoded(torch, stopping_encoded, target_device)
    stopping_labels_tensor = torch.tensor(
        stopping_labels, dtype=torch.float32, device=target_device
    )
```

Inside the per-epoch mini-batch loop, replace the single `features_tensor`/`batch_features` indexing with per-group indexing and the model call:

```python
    train_size = labels_tensor.shape[0]
    recent_stopping_losses: deque[float] = deque(maxlen=_STOPPING_SMOOTHING_WINDOW)
    for epoch in range(1, epochs + 1):
        model.train()
        permutation = torch.randperm(train_size, device=target_device)
        epoch_loss_total = 0.0
        for start in range(0, train_size, batch_size):
            batch_indices = permutation[start : start + batch_size]
            optimizer.zero_grad()
            probabilities = model(
                blue_picks_tensor[batch_indices],
                blue_bans_tensor[batch_indices],
                red_picks_tensor[batch_indices],
                red_bans_tensor[batch_indices],
                continuous_tensor[batch_indices],
            )
            loss = torch.nn.functional.binary_cross_entropy(
                probabilities, labels_tensor[batch_indices]
            )
            loss.backward()
            optimizer.step()
            epoch_loss_total += float(loss.detach()) * batch_indices.shape[0]
        train_loss = epoch_loss_total / train_size
        if not stopping_encoded:
            if verbose:
                print(f"Epoch {epoch}/{epochs}: train_loss={train_loss:.4f}")
            continue
        model.eval()
        with torch.inference_mode():
            stopping_loss = float(
                torch.nn.functional.binary_cross_entropy(
                    model(*stopping_tensors), stopping_labels_tensor
                )
            )
```

(The rest of the early-stopping logic below this point — `recent_stopping_losses.append(...)`, `smoothed_stopping_loss`, `is_best`, `best_state`, the verbose prints, and the `patience` break — is unchanged; it only consumed `stopping_loss`/`train_loss` by value, not the tensors directly.)

Update the final validation-metrics block:

```python
    model.eval()
    if validation_encoded:
        with torch.inference_mode():
            validation_tensors = _stack_encoded(torch, validation_encoded, target_device)
            model_probabilities = [
                float(value) for value in model(*validation_tensors).detach().cpu().tolist()
            ]
    else:
        model_probabilities = []
```

And the checkpoint save block:

```python
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.cpu().state_dict(),
            "num_champions": feature_space.num_champions,
            "continuous_size": feature_space.continuous_size,
            "embedding_dim": 16,
            "hidden_size": hidden_size,
            "dropout": dropout,
            "feature_space": feature_space.model_dump(mode="json"),
        },
        output_dir / "model.pt",
    )
```

- [ ] **Step 5: Update `predictor.py`**

Replace the body of `WinPredictor.load` and `WinPredictor.predict` in `backend/app/model/predictor.py`:

```python
    @classmethod
    def load(cls, path: Path) -> "WinPredictor":
        """Load a local model checkpoint and its serialized feature space.

        Uses GPU (CUDA) when available, falling back to CPU otherwise.
        """
        torch = _torch_module()
        from app.model.net import DraftNet

        device = "cuda" if torch.cuda.is_available() else "cpu"
        checkpoint = torch.load(path, map_location=device, weights_only=True)
        if not isinstance(checkpoint, dict):
            raise ValueError(f"Invalid model checkpoint: {path}")
        try:
            feature_space = FeatureSpace.model_validate(checkpoint["feature_space"])
            num_champions = int(checkpoint["num_champions"])
            continuous_size = int(checkpoint["continuous_size"])
            embedding_dim = int(checkpoint["embedding_dim"])
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
            embedding_dim=embedding_dim,
            hidden_size=hidden_size,
            dropout=dropout,
        )
        model.load_state_dict(state_dict)
        model.eval()
        model = model.to(device)
        predictor = cls(model, feature_space)
        predictor._device = device
        return predictor

    def predict(self, states: Sequence[DraftState]) -> list[float]:
        """Predict blue-side game win probabilities for a batch of draft states.

        Uses GPU (CUDA) when available for faster inference.
        """
        if not states:
            return []
        torch = _torch_module()
        encoded = [self.feature_space.encode(state) for state in states]
        blue_picks = torch.tensor(
            [state.blue_picks for state in encoded], dtype=torch.long, device=self._device
        )
        blue_bans = torch.tensor(
            [state.blue_bans for state in encoded], dtype=torch.long, device=self._device
        )
        red_picks = torch.tensor(
            [state.red_picks for state in encoded], dtype=torch.long, device=self._device
        )
        red_bans = torch.tensor(
            [state.red_bans for state in encoded], dtype=torch.long, device=self._device
        )
        continuous = torch.tensor(
            [state.continuous for state in encoded], dtype=torch.float32, device=self._device
        )
        with torch.inference_mode():
            predictions = self._model(blue_picks, blue_bans, red_picks, red_bans, continuous)
        if self._device != "cpu":
            predictions = predictions.cpu()
        return [float(probability) for probability in predictions.tolist()]
```

- [ ] **Step 6: Run the two target tests to verify they pass**

Run: `cd backend && uv run pytest tests/model/test_training.py::test_training_learns_planted_winning_champion_and_loads_batch_predictor tests/model/test_training.py::test_train_model_saves_embedding_checkpoint_fields -v`
Expected: Both PASS.

- [ ] **Step 7: Run the full backend gate and fix any remaining fallout**

Run: `cd backend && uv run ruff check . && uv run ruff format --check . && uv run basedpyright && uv run pytest -q`
Expected: All green. If any other existing test in `tests/model/test_training.py` still references the old `hidden_size`/`dropout`-only checkpoint shape, `DraftNet(input_size=...)`, or passes a single tensor into `model(...)`, update that test's body to match this task's new shapes (same pattern as Steps 1-6) before considering the task done — do not leave any test red.

- [ ] **Step 8: Commit**

```bash
git add backend/app/model/train.py backend/app/model/predictor.py backend/tests/model/test_training.py
git commit --trailer "Closes: #<ISSUE_NUMBER>" -m "feat: wire training and inference to champion-embedding input shape"
```

---

### Task 4: Down-weight near-empty draft prefixes

**Files:**
- Modify: `backend/app/model/train.py`
- Test: `backend/tests/model/test_training.py`

**Interfaces:**
- Consumes: `_PrefixExample` dataclass and `_prefix_examples` (existing, this task adds a field and a computation to them); `DRAFT_ORDER` (existing import, used for `len(DRAFT_ORDER)`).
- Produces: `_PrefixExample.weight: float` field (`prefix_length / len(DRAFT_ORDER)`, so `0.0` for an empty draft up to `1.0` for a complete one). Training and stopping loss computations become weighted means using this field.

- [ ] **Step 1: Write the failing test for the weight computation**

```python
def test_prefix_examples_weight_scales_with_locked_in_actions(make_game: Any) -> None:
    game = _planted_game(make_game, 0, True)

    examples = _prefix_examples([game])

    weights_by_prefix_length = {len(example.state.actions): example.weight for example in examples}
    assert weights_by_prefix_length[0] == 0.0
    assert weights_by_prefix_length[len(DRAFT_ORDER)] == 1.0
    assert weights_by_prefix_length[10] == 10 / len(DRAFT_ORDER)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && uv run pytest tests/model/test_training.py::test_prefix_examples_weight_scales_with_locked_in_actions -v`
Expected: FAIL with `AttributeError: '_PrefixExample' object has no attribute 'weight'`.

- [ ] **Step 3: Add the `weight` field and compute it in `_prefix_examples`**

```python
@dataclass(frozen=True)
class _PrefixExample:
    game: GameDraft
    state: DraftState
    label: float
    weight: float
```

```python
def _prefix_examples(games: list[GameDraft]) -> list[_PrefixExample]:
    examples: list[_PrefixExample] = []
    for game in games:
        if not _is_decided(game):
            continue
        label = 1.0 if game.winner_team_id == game.blue_team_id else 0.0
        action_ids = [action.champion_id for action in game.actions[: len(DRAFT_ORDER)]]
        for prefix_length in range(len(action_ids) + 1):
            examples.append(
                _PrefixExample(
                    game=game,
                    state=DraftState(
                        patch=game.patch or "unknown",
                        actions=action_ids[:prefix_length],
                    ),
                    label=label,
                    weight=prefix_length / len(DRAFT_ORDER),
                )
            )
    return examples
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && uv run pytest tests/model/test_training.py::test_prefix_examples_weight_scales_with_locked_in_actions -v`
Expected: PASS.

- [ ] **Step 5: Write the failing test for weighted-loss training**

```python
def test_train_model_uses_weighted_loss_without_crashing(make_game: Any, tmp_path: Path) -> None:
    games = [_planted_game(make_game, index, index % 2 == 0) for index in range(24)]

    report = train_model(games, tmp_path, epochs=3, hidden_size=8, seed=17, device="cpu")

    assert report.metrics["model"]["auc"] is not None
```

This mainly guards against a shape/broadcast crash when the weight tensor is introduced into the loss computation; it will pass once Step 7 below is implemented correctly, and fail (with a tensor shape or `KeyError` on `example.weight`) if the implementation is wrong.

- [ ] **Step 6: Run test to verify it fails before the loop changes**

Run: `cd backend && uv run pytest tests/model/test_training.py::test_train_model_uses_weighted_loss_without_crashing -v`
Expected: PASS already, since Step 3 alone doesn't touch the training loop yet — this is expected; the real regression guard is the next step's gate run. Proceed to Step 7 regardless; this test exists to catch a broadcasting mistake introduced there.

- [ ] **Step 7: Thread the weight through the training and stopping loss**

In `train_model`, after building `train_labels`/`stopping_labels`, also build the weight lists:

```python
    train_labels = [example.label for example in train_examples]
    train_weights = [example.weight for example in train_examples]
```

```python
    stopping_labels = [example.label for example in stopping_examples]
    stopping_weights = [example.weight for example in stopping_examples]
```

Build the corresponding tensors alongside `labels_tensor`/`stopping_labels_tensor`:

```python
    labels_tensor = torch.tensor(train_labels, dtype=torch.float32, device=target_device)
    weights_tensor = torch.tensor(train_weights, dtype=torch.float32, device=target_device)
```

```python
    stopping_labels_tensor = torch.tensor(
        stopping_labels, dtype=torch.float32, device=target_device
    )
    stopping_weights_tensor = torch.tensor(
        stopping_weights, dtype=torch.float32, device=target_device
    )
```

Replace the per-batch loss computation:

```python
            batch_weights = weights_tensor[batch_indices]
            per_example_loss = torch.nn.functional.binary_cross_entropy(
                probabilities, labels_tensor[batch_indices], reduction="none"
            )
            loss = (per_example_loss * batch_weights).sum() / batch_weights.sum().clamp_min(1e-8)
```

(This replaces the single-line `loss = torch.nn.functional.binary_cross_entropy(probabilities, labels_tensor[batch_indices])` call from Task 3 Step 4 — it must come after `probabilities = model(...)` and before `loss.backward()`.)

Replace the stopping-loss computation:

```python
        model.eval()
        with torch.inference_mode():
            stopping_probabilities = model(*stopping_tensors)
            stopping_per_example_loss = torch.nn.functional.binary_cross_entropy(
                stopping_probabilities, stopping_labels_tensor, reduction="none"
            )
            stopping_loss = float(
                (stopping_per_example_loss * stopping_weights_tensor).sum()
                / stopping_weights_tensor.sum().clamp_min(1e-8)
            )
```

- [ ] **Step 8: Run both new tests plus the full training test file**

Run: `cd backend && uv run pytest tests/model/test_training.py -v`
Expected: All pass, including `test_train_model_uses_weighted_loss_without_crashing` and `test_prefix_examples_weight_scales_with_locked_in_actions`.

- [ ] **Step 9: Run the full backend gate**

Run: `cd backend && uv run ruff check . && uv run ruff format --check . && uv run basedpyright && uv run pytest -q`
Expected: All green.

- [ ] **Step 10: Commit**

```bash
git add backend/app/model/train.py backend/tests/model/test_training.py
git commit --trailer "Closes: #<ISSUE_NUMBER>" -m "feat: down-weight near-empty draft prefixes in training loss"
```

---

### Task 5: Ablation retrain and PR writeup

**Files:**
- None (this task is verification and documentation, not code).

**Interfaces:**
- Consumes: the real GRID dataset via `uv run python -m app.pipeline.build` and `uv run python -m app.model.train` (both already support `--out`/`STL_DATA_DIR`/`DRAFT_COACH_DERIVED_DIR` per existing CLI).

- [ ] **Step 1: Run the full backend gate one more time on the final branch state**

Run: `cd backend && uv run ruff check . && uv run ruff format --check . && uv run basedpyright && uv run pytest -q`
Expected: All green.

- [ ] **Step 2: Retrain on real GRID data and record metrics**

Run: `uv run python -m app.model.train` (from `backend/`, with `STL_DATA_DIR` pointed at a real GRID data directory and `backend/.derived/drafts.json` already built via `uv run python -m app.pipeline.build`).
Record the resulting `backend/.derived/model_metrics.json`'s `metrics.model.auc`/`accuracy`/`log_loss` values.

- [ ] **Step 3: Compare against the 0.515 AUC baseline and the mini-batch/regularization-only numbers from this session**

Write a short before/after table for the PR description: baseline (0.515 AUC, pre-Milestone-1), mini-batch + regularization only (this session's earlier retrains), and this plan's embeddings + prefix-weighting result. State plainly whether AUC improved, and if it didn't, say so — the exit criteria only requires documenting the result, not that every change helped.

- [ ] **Step 4: Open the PR**

Push the branch and open a PR against `main` referencing the issues closed by Tasks 1-4, with the before/after metrics table from Step 3 in the PR body. Do not push or open the PR without the user's explicit go-ahead, per this repository's git-push convention already established earlier in this project's history.
