# Recommendation Engine — Implementation Plan

**Issue:** #9 — recommendation engine (PyTorch NN fitness + GA)
**Roadmap step:** 3 — recommendation engine
**Design:** `docs/superpowers/specs/2026-11-19-recommendation-engine-design.md`

## Goal

Train a win-probability network offline on local GRID drafts and use it as the fitness function
of a rest-of-phase genetic search that drives `POST /draft/recommend`, keeping the statistical
scorer as the explanation layer and as the fallback when no model exists.

## Touch points

- `backend/app/pipeline/meta.py`, `drafts.py`, `dataset.py`: `started_at`, `PlayerLine`,
  `extract_game_lineups`, lineups on `GameDraft` (defaults keep older fixtures valid).
- `backend/app/pipeline/ddragon.py` + `build.py`: best-effort Data Dragon cache, `--no-ddragon`.
- `backend/app/model/features.py`, `net.py`, `train.py`, `predictor.py`.
- `backend/app/engine/ga.py`; `service.py` and `models.py` (`engine` response field).
- `backend/pyproject.toml` / `uv.lock`: `cpu` and `cu128` torch extras on PyTorch indexes.
- `.gitignore`: model weights and metrics; `README.md`, `CLAUDE.md`.

## Test-first sequence

1. Lineups and `started_at` in the pipeline.
2. Data Dragon parse, fetch/cache, offline fallback; build flag.
3. Feature space fit/encode/serialize.
4. Net, time split, training on a planted signal, predictor reload.
5. GA exhaustive and genetic paths: legality, replies, ranking, determinism.
6. Service and API: model path, side orientation, statistical fallback, `engine` field.

## Safeguards

- Synthetic fixtures only; no real feed rows in tests.
- `model.pt`, `model_metrics.json`, `ddragon.json` and all derived files stay in the ignored
  derived directory; `*.pt`, `*.pth` and `model_metrics.json` are ignored everywhere.
- Stage files explicitly and check `git diff --cached --name-only` before committing.

## Out of scope

Frontend, deployment, deeper-than-phase lookahead, external meta badge.
