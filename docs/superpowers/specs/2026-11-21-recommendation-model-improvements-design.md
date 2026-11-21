# Recommendation Model Improvements — Design

Supersedes the training result in `2026-11-19-recommendation-engine-design.md`: that
spec's held-out run (seed 42, 1,418 training games) measured accuracy 0.553 / AUC 0.515 /
log loss 0.692 for the model, against 0.554 / 0.500 / 0.689 for a constant blue-side-rate
baseline. **AUC 0.515 means the model cannot distinguish a winning draft from a losing one
better than a coin flip**, and its accuracy is below the baseline's. This is the confirmed
root cause of recommendations looking arbitrary on real matchups (e.g. T1 vs. BLG): the
genetic algorithm's fitness function is effectively noise, so a longer search explores that
noise more thoroughly rather than finding better picks. Fixing the model comes before any
change to how long the search runs.

## Goal

Raise the model's held-out AUC well above 0.5 (target ballpark 0.58–0.65, in line with
published draft-only win-prediction results — draft alone never fully predicts a game) using
both better modeling and more training data, while keeping a clean before/after number at
every step so we know which change caused which effect. Live/background search ("keep
improving until the user acts") and any related UI/UX changes are explicitly **out of scope**
for this spec — parked as a follow-up once the model is trustworthy.

## Baseline (Milestone 0 — captured, not redone)

From `backend/.derived/model_metrics.json`:

| | accuracy | AUC | log loss |
|---|---|---|---|
| model | 0.553 | 0.515 | 0.692 |
| blue-rate baseline | 0.554 | 0.500 | 0.689 |
| statistical scorer | 0.547 | 0.527 | 0.689 |

`train_games=1418`, `validation_games=316`, `stopping_games=145`, `best_epoch=7` (of 100
allowed) — stopping this early is itself a symptom that the model found nothing to learn
before its stopping-set loss started getting worse.

## Milestone 1 — fix the model on existing GRID data

No new data sources yet, so any AUC movement here is attributable only to these four
changes. Each is retrained and measured independently (ablation: toggle one at a time) before
they're combined, so the PR can report which change did what.

1. **Per-role counter-matchup feature.** Today `scoring.py` has same-team synergy (do two of
   our own picks work well together) but nothing for "how does my pick do against their pick
   in the same role." Add a smoothed per-role champion-vs-champion win-rate signal (same
   smoothing pattern as existing edges in `scoring.py`: pulled toward 50% when a pair has few
   games), and a parallel feature block in `features.py` for the NN. Surfaced in the
   plain-language "why" the same way existing ours/theirs factors are, per Product Principle
   2 (every recommendation shows its why).
2. **Blue/red flip augmentation.** In `train.py`'s example generation, add a mirrored copy of
   every example with blue/red swapped and the label flipped. Doubles effective training data
   for free and removes any blue-side bias the model could otherwise learn instead of real
   draft strength.
3. **Champion embeddings instead of one-hot presence.** `features.py`'s current encoding is a
   4-channel one-hot over every champion — mostly empty space for ~1,400 games, a classic
   high-dimensional/low-data setup that invites memorization over generalization. Replace with
   a learned embedding per champion. This is the largest code change in this milestone: it
   changes `net.py`'s input shape from a flat precomputed vector to champion-index lookups plus
   the existing aggregate features, and `predictor.py`'s inference path needs to build the same
   input shape.
4. **Down-weight near-empty draft prefixes.** `_prefix_examples` in `train.py` currently trains
   on every prefix (0–20 actions) with the game's real final label, including prefixes with 0–3
   picks made that can't realistically predict the outcome. Weight examples by how many picks
   are locked in (or drop the earliest prefixes below a small threshold) so the loss isn't
   dominated by near-random early states.

**Exit criteria for Milestone 1:** retrain, compare each toggle's AUC against the 0.515
baseline, document which changes helped in the PR description, and decide the combination that
ships before moving to Milestone 2.

## Milestones 2–4 — widen the training data

Each source gets its own small parser that normalizes into the *same* internal shape the GRID
pipeline already produces (`GameDraft` / `DraftAction` / `PlayerLine` from
`app/pipeline/dataset.py` / `drafts.py`), so `features.py` and `train.py` don't need to know
where a game came from. Each is a manual build step (same pattern as
`uv run python -m app.pipeline.build`), writing into the existing gitignored `.derived/`
output — raw downloaded/scraped files are never committed, same rule as the GRID data (see
`AGENTS.md`).

- **Milestone 2 — Oracle's Elixir.** Free downloadable CSVs of pro matches; closest fit to the
  GRID shape (picks, bans, player stats, win/loss, patch). New module, e.g.
  `app/pipeline/sources/oracles_elixir.py`.
- **Milestone 3 — Leaguepedia.** Pro and amateur tournament history via Leaguepedia's official
  Cargo query API (not raw page scraping — respects their terms, more stable). Adds regional/
  amateur coverage GRID and Oracle's Elixir may lack.
- **Milestone 4 — Riot Developer API.** Different in kind: per-champion high-elo (Diamond+)
  solo-queue pick/win/ban rate by patch, not full 10-pick coordinated drafts. Refreshes the
  existing "meta edge" strength signal only — **not** added to the win-outcome training set,
  since solo-queue games aren't comparable draft states. Needs a personal developer API key;
  Riot's terms forbid storing/redistributing raw match data, so only aggregated rates are
  cached locally, same derived-artifact pattern as everything else.

**Cross-source handling (applies to all three):**

- Missing fields (e.g. a source without vision score) default to neutral/zero, same coercion
  pattern `drafts.py`'s `_number()` already uses.
- No attempt at perfect dedup across sources — best-effort match on (date, teams, patch),
  logged rather than fatal, documented as a known limitation.
- Each milestone retrains and reports its own before/after AUC against the Milestone 1 result.

## Milestone 5 — final validation

- Compare final accuracy/AUC/log loss against the Milestone 0 baseline table above.
- Manually re-run T1 vs. BLG and 1–2 other real matchups through the app; confirm the picks
  look sensible (the originally reported problem).
- Update `PRODUCT.md`/this spec with the final data source list and metrics in the same PR,
  per `CONVENTIONS.md`.

## Testing

TDD per `CONVENTIONS.md`, mirroring the existing test layout:

- New source parsers: `backend/tests/pipeline/test_<source>.py`, hand-written synthetic
  fixtures mimicking each source's real format (fake CSV rows / fake Cargo API JSON / fake
  Riot API JSON) — never real downloaded or scraped data as a fixture.
- New model logic (matchup smoothing, augmentation, embedding encoding, prefix weighting):
  `backend/tests/model/`, mirroring `app/model/`.
- One end-to-end check: `train_model()` runs on a small fabricated multi-source dataset without
  crashing and produces a normal `TrainingReport`.
- The ablation comparison (Milestone 1) is a one-off analysis script, not part of the pytest
  suite — its job is to produce a number for the PR description, not to guard against
  regressions.
- Existing gates (`ruff check`, `ruff format --check`, `basedpyright`, `pytest -q`) stay green
  through every milestone; one PR per milestone.
- Since `model_metrics.json` lives in the gitignored `.derived/` directory and is never
  committed, each milestone's PR description carries a short before/after metrics table so
  there's a durable record in PR history even though the artifact itself isn't tracked.

## Explicitly out of scope (parked for later specs)

- Live/background search that keeps improving a displayed recommendation until the user picks,
  bans, or undoes (needs a cancellable background job and a way to push updates to the
  browser — the current stateless request/response API doesn't support this yet).
- Any UI/UX changes, including how new reasoning factors (e.g. counter-matchup) are displayed
  beyond the existing "why" text pattern, and how the live-search feature above would be shown.
- Switching away from the PyTorch neural net to a tree-based model (e.g. XGBoost) — the product
  docs describe the recommendation engine as NN + GA for the hackathon pitch; this spec keeps
  that and focuses improvement effort on features/data/training instead.
