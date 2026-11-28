# AUC-Based Early Stopping — Design

Context: two experiments on this branch (feeding the explainable statistical scorer's
probability into `FeatureSpace.encode()`, then replacing `DraftNet`'s mean-pooling with
order-aware self-attention over the draft sequence) were implemented correctly — reviewed
clean, no bug found in either diff — and both were reverted anyway, because retraining after
each collapsed validation AUC from the current best (0.588) to exactly 0.500: the model
degenerated to predicting the constant majority class. `best_epoch` dropped from 22 (current
baseline) to 8 in both cases.

The likely root cause: `train_model` (`backend/app/model/train.py`) selects its "best"
checkpoint by the **lowest smoothed BCE loss** on a held-out stopping set, not by AUC. With a
weak signal-to-noise ratio (this project's recorded ceiling — see
`docs/superpowers/specs/2026-11-19-recommendation-engine-design.md`), confidently predicting
the training-set base rate is often enough to *minimize loss* even with zero discriminative
power. Any change that nudges the loss landscape — including a correctly-implemented, useful
feature — can make that trivial local minimum reachable earlier than real signal, and loss-based
stopping has no way to tell the difference between "found a lower loss because it generalizes
better" and "found a lower loss because it gave up and predicts the prior." AUC does not have
this blind spot: a constant predictor always scores AUC 0.5, so an AUC-based criterion can never
mistake collapse for progress.

## Goal

Change `train_model`'s early-stopping checkpoint selection from loss-based to AUC-based, verify
it on the current (already-reverted) baseline, then re-apply the two reverted changes on top of
it one at a time to see whether the fix was in fact the missing piece. Each step gets its own
commit and its own before/after AUC measurement, so if something still regresses we know exactly
which change caused it.

## Part 1 — AUC-based early stopping

**File:** `backend/app/model/train.py` only (plus its tests in `backend/tests/model/test_training.py`).

**Mechanism:** The per-epoch loop already computes `stopping_probabilities` and
`stopping_labels_tensor` each epoch to derive `stopping_loss` via a smoothed
`recent_stopping_losses: deque[float]` window (`_STOPPING_SMOOTHING_WINDOW = 3`). Add a parallel
smoothed AUC computed from the same `stopping_probabilities`/`stopping_labels` via the existing
`_metrics()` helper (which already returns `auc: float | None`). Checkpoint selection changes
from "is this epoch's smoothed loss the lowest so far" to "is this epoch's smoothed AUC the
highest so far, tie-breaking on lower smoothed loss." Patience and the rest of the early-stop
control flow are unchanged — only the definition of "best" changes.

**Error handling:** `_metrics()` returns `auc=None` when the stopping set has no positives or no
negatives (small or degenerate stopping split). Precise rule, so AUC-defined and AUC-undefined
epochs are never compared against each other:
- Track `best_auc: float | None` alongside `best_loss`.
- An epoch with a defined smoothed AUC can become "best" if `best_auc` is `None` (no epoch has
  had a defined AUC yet) or its AUC beats `best_auc` (ties broken on lower smoothed loss) — and
  doing so sets `best_auc` to a real value.
- An epoch with `auc=None` can become "best" only while `best_auc` is still `None` (i.e. no
  epoch so far, including this one, has ever had a defined AUC), compared on smoothed loss
  alone. Once any epoch sets a real `best_auc`, later `None`-AUC epochs are never selected.
- Net effect: if the stopping set is persistently degenerate for the whole run, this degrades
  exactly to today's loss-only behavior; otherwise, once real AUC signal appears, it always
  wins. This never raises and never stalls the training loop.

**Testing:** TDD per `CONVENTIONS.md`. Cover: (1) a case where loss keeps improving slightly
epoch-over-epoch while AUC does not — asserting the selected checkpoint is the one with the best
AUC, not the lowest loss (the exact failure mode this fix targets); (2) the `auc=None` fallback
path does not raise and still selects a checkpoint; (3) the existing patience/early-stop
behavior (number of epochs run, early termination) is unchanged by this fix on data where loss
and AUC move together. These can be built as narrow unit tests around the stopping-loop logic
with a fabricated/mocked sequence of per-epoch probabilities, without requiring a full real
training run.

**Verification:** retrain on the real local dataset (`uv run python -m app.model.train`) on the
current reverted baseline and compare against the recorded 0.588 AUC. Expected: neutral to
slightly positive, since the current baseline already trains past epoch 8 — if this step alone
regresses the baseline, stop and report rather than proceeding to re-apply anything.

## Part 2 — re-apply the two reverted changes

Only proceeds if Part 1's verification does not regress the 0.588 baseline.

1. Re-apply the scorer-probability feature: fold-safe `aggregates: dict[str, PatchAggregate]`
   field on `FeatureSpace` (fit from `train_games`), a cached per-patch `BlendedStats` lookup,
   and a `_scorer_probability()` helper appended as the last element of `EncodedState.continuous`
   — identical design to the reverted `934b9dd`/`8cfad8f` commits (available via
   `git show 934b9dd`/`git show 8cfad8f` as a reference, not a plan to copy verbatim without
   re-verifying against the current file state). Retrain, compare against Part 1's result.
2. Re-apply order-aware self-attention pooling: `EncodedState.draft_sequence` replacing the four
   slot-group arrays, `DraftNet` gaining a position embedding and one self-attention block in
   place of the 4-group mean pool — identical design to the reverted `6275106` commit. Retrain,
   compare against step 1's result.

Each step is its own commit, its own retrain, and its own recorded AUC — same discipline as the
reverted attempt, so a regression at either step is immediately attributable.

## Testing (overall)

- Existing gates (`uv run ruff check .`, `uv run ruff format --check .`, `uv run basedpyright`,
  `uv run pytest -q`, all from `backend/`) stay green after every step.
- TDD mandatory: write the failing test before the implementation in every step.
- `model_metrics.json` lives in the gitignored `.derived/` directory, so each step's before/after
  AUC is recorded in its commit message and in the implementation plan's progress ledger, not
  relied on as a committed artifact.

## Explicitly out of scope

- Any change to the statistical scorer (`app/engine/scoring.py`) itself, or to live serving
  (`app/engine/service.py`, `app/engine/ga.py`) — this spec is training-time only.
- A general hyperparameter re-sweep (`app/model/tune.py`) of dropout/weight_decay/learning_rate —
  worth considering later if Part 1 alone doesn't fully resolve the collapse pattern, but not
  part of this spec; Part 1's fix is a training-loop correctness fix, not a tuning pass.
- Adding a runtime ablation flag for either reverted feature — git history remains the ablation
  axis, per this project's established convention.
