# Data Pipeline Foundation — Implementation Plan

**Issue:** #5 — `[Pipeline] Build backend data pipeline foundation`
**Roadmap step:** 1 — data pipeline

## Goal

Build the tested backend foundation that reads the local GRID dataset, extracts per-game drafts,
computes patch-bucketed aggregates, and writes derived JSON files locally. Raw data and derived
artifacts must never be committed or pushed.

## Decisions

- Use Python 3.12 with `uv` and `zstandard` for streaming `.jsonl.zst` feeds.
- Resolve the data directory from `STL_DATA_DIR` when set; otherwise try the project-local
  `stl-hackathon-2026-grid-starter/`, then the sibling `../stl-hackathon-2026-grid-starter/`.
  Fail clearly if the selected directory does not contain `meta/`.
- Write Pydantic v2 JSON outputs to ignored `backend/.derived/`: `drafts.json` and
  `aggregates.json`.
- Keep Python package layout flat under `backend/app/`; tests mirror it under `backend/tests/`.

## Scope and touch points

- `backend/pyproject.toml` and `backend/uv.lock`: Python 3.12 dependencies and Ruff, basedpyright,
  and pytest configuration.
- `backend/app/config.py`: data-directory resolution and settings.
- `backend/app/pipeline/meta.py`: Pydantic models and LoL CSV loaders for games, champions, teams.
- `backend/app/pipeline/drafts.py`: final-state streaming and `draftActions` extraction.
- `backend/app/pipeline/dataset.py`: join drafts with game metadata; skip unknown games and games
  without drafts.
- `backend/app/pipeline/aggregates.py`: patch aggregates and recent-patch blending.
- `backend/app/pipeline/build.py`: CLI writing both JSON outputs.
- `backend/tests/`: config, metadata, draft extraction, dataset, aggregates, and build tests;
  fixtures are synthetic and hand-written/generated in test temporary directories.
- `.gitignore`: exclude `backend/.derived/` and Python/test tool caches.

## Test-first implementation sequence

For each task, write the focused test first, run it to confirm the expected failure, implement the
smallest change, then rerun the test and relevant suite.

1. **Scaffold and configuration**
   - Test `STL_DATA_DIR` override, local/sibling candidate fallback, and clear failure when `meta/`
     is absent.
   - Add `pyproject.toml`, package initializers, settings/configuration, and required ignore rules.
2. **Metadata loaders**
   - Use synthetic CSV fixtures to test typed parsing, filtering to `title=lol`, and side mapping.
   - Implement `GameMeta`, `Champion`, `Team` and `load_games`, `load_champions`, `load_teams`.
3. **Draft extraction**
   - Generate a tiny `.jsonl.zst` feed in `tmp_path`; test that envelopes without state are skipped,
     the last `seriesState` wins, and actions are ordered by sequence.
   - Implement `DraftAction`, `read_final_series_state`, and `extract_game_drafts`.
4. **Dataset join**
   - Test joining synthetic game metadata to extracted drafts and skipping unknown games or games
     without drafts.
   - Implement `GameDraft` and `build_game_drafts` using `lol/series.csv` to locate feeds.
5. **Aggregates**
   - Test champion pick/ban/win counts, blue/red side wins, unordered same-team champion-pair
     synergy, team/champion history, and recent-patch blending.
   - Sort patch versions numerically (`15.10` newer than `15.9`); blend patches until the configured
     minimum-games threshold is met. Exclude unfinished games or games without a winner from win
     statistics, including their denominators (`decided_games`, `decided_picks`, side and pair
     game counts); pick/ban presence still counts every game. Blank or non-numeric patch values
     (2 blank rows in the real index) go to an `unknown` bucket that sorts older than every
     numeric patch, so it is only blended in as a last resort.
6. **Build command**
   - Test an end-to-end synthetic data directory and verify both output files are written.
   - Implement `python -m app.pipeline.build [--out]`, serializing per-game drafts and aggregates.

## Data and validation safeguards

- Never use real GRID rows or feed excerpts in tests; construct synthetic fixtures.
- Stream compressed feeds line by line; do not load a full feed into memory.
- A real-data smoke run may report counts only. Keep the data directory and all outputs local and
  ignored.
- Run targeted pytest tests as each task lands, then the full backend pytest suite, Ruff, and
  basedpyright.
- Before any commit, inspect `git diff --cached --name-only` and ensure no data or derived artifact
  is staged.

## Out of scope

FastAPI endpoints, Fearless draft state machine, statistical explanation scorer, PyTorch NN/GA, and
frontend work belong to later roadmap plans.
