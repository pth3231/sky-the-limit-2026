# Recommendation Engine (NN + GA) — Design

Roadmap step 3. Refines section 4 of
`2026-11-19-lol-draft-coaching-assistant-design.md`.

## Goal

Replace the statistical scorer as the *ranking* and *win-rate* source with a PyTorch
win-probability network used as the fitness function of a genetic algorithm. The statistical
`Scorer` stays as the plain-language "why" and as the fallback engine when no trained model is
available.

## Data inputs (all local, never committed)

- `drafts.json` from the pipeline, now also carrying each game's `started_at` and complete
  five-player lineups per team (`PlayerLine`: role, champion, damage share, damage taken,
  vision score, gold share, kill participation). Roles come from the player's `position`
  when present, otherwise from player order (top, jungle, mid, bot, support).
- Optional `ddragon.json`: Data Dragon champion tags and `info` ratings, fetched best-effort
  at build time. A failed download falls back to the previous cache or to no traits; the
  build never fails because of it. `--no-ddragon` skips the download entirely.

## Features (`app/model/features.py`)

`FeatureSpace.fit(train_games, ddragon)` is fitted on training games only and serialized
inside the model checkpoint. During training, `vocabulary_games` can keep champion, patch, and
trait dimensions fixed across folds while role assignments and historical profiles are fitted
only from each fold's history games. `encode(DraftState)` produces a fixed-width, blue-oriented
vector:

- pick/ban presence per side (4 × champions; unseen champions add nothing);
- one-hot action count (0–20) and patch (training patches + "other");
- per side: role mix of picked champions, average role win rate / strength / GRID performance
  profile, Data Dragon tag counts and ratings, and average patch pick/ban/win rates for picks
  and for bans.

## Model and training (`app/model/net.py`, `app/model/train.py`)

- MLP: two hidden ReLU layers with dropout, sigmoid output = blue-side game win probability.
- Examples: every draft prefix (0–20 actions) of every decided game, labelled by the winner.
- Split: chronological by series `started_at`, with the newest 15% of series held out. Series
  stay intact across outer validation and early-stopping splits; unknown timestamps are ordered
  deterministically. Seeded; device `auto|cpu|cuda`.
- Training examples use deterministic, series-grouped out-of-fold history (up to five folds).
  Champion, patch, and trait vocabularies are fixed from the fitting window, while historical
  pick/ban/win rates, role assignments, and GRID player profiles are fit outside each example's
  fold. With one series, history is empty rather than self-derived. Stopping and validation
  features use a feature space fitted only on the earlier fitting window.
- Early stopping uses the newest 15% of the training window, Adam weight decay, and keeps the
  best stopping-set epoch's weights.
- Leakage-safe real-data run (seed 42, 1,418 games in the training window / 316 held out / 145
  stopping): held-out accuracy / AUC / log loss were 0.553 / 0.515 / 0.692 for the model,
  0.554 / 0.500 / 0.689 for the blue-rate baseline, and 0.547 / 0.527 / 0.689 for the
  statistical scorer. The model did not clearly improve on these baselines; the metrics are
  evidence for this data snapshot, not a guarantee of prediction quality.
- Metrics on the held-out window (accuracy, AUC, log loss) for the model, a constant
  blue-side-rate baseline, and the statistical scorer, written to `model_metrics.json`.
- Artifacts: `model.pt` (weights + feature space) and `model_metrics.json` in the derived
  directory, both git-ignored.

## Search (`app/engine/ga.py`)

- Genome: our remaining slots in the current phase (bans or picks until the action type
  changes). Opponent slots in that window are filled with their most likely legal reply:
  their own pick/ban history first, then global presence, then the first legal champion.
- Legality: every simulated action excludes already-used and Fearless/unavailable champions.
- Small spaces (≤ 256 permutations) are searched exhaustively; otherwise tournament selection,
  uniform crossover, mutation and repair, with elitism. The RNG is seeded from a hash of the
  request, so identical requests return identical answers.
- Output: distinct first moves ranked by the best model score found for each.

## Service integration

- `load_engine` loads `model.pt` when present and PyTorch is importable; otherwise the engine
  stays statistical.
- With a model: game win rate is the model's prediction for the current state (flipped for red
  side), candidates come from the search, and each candidate keeps the scorer's contributing
  factors. The reason states the model estimate and the top statistical signal, so the UI never
  implies the factors are the model's internals.
- `DraftResponse.engine` is `"model"` or `"statistical"`.

## Dependencies

PyTorch is optional: `uv sync --extra cpu` or `uv sync --extra cu128` (CUDA 12.8 wheels,
needs an NVIDIA driver ≥ 570). The extras conflict, so only one can be installed.

## Testing

Synthetic fixtures only: lineup extraction and role mapping; Data Dragon parsing, caching and
offline fallback; feature fitting on training games only, orientation, stable fold vocabulary,
series-grouped splits, own-series outcome/performance isolation, deterministic and single-series
out-of-fold behavior; training learns a planted winning champion and reloads; GA legality,
opponent replies, best-move ranking and determinism; service model path, orientation and
statistical fallback.
