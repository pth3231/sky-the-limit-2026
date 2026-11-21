# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

**Always read [AGENTS.md](AGENTS.md) at the start of any session in this repository and follow its rules.** It contains binding instructions for AI agents (notably the data-handling and licensing rules) that take precedence over the rest of this file.

## Status

The repository contains a Python backend in `backend/`: a streamed GRID data pipeline,
patch aggregates, a stateless FastAPI draft API, Fearless draft validation, an explainable
statistical scorer, and an optional PyTorch win-probability model used as the fitness function of
a genetic search (statistical fallback when no model is present). The Next.js frontend is
planned; see [README.md](README.md#roadmap) for the roadmap.

Run backend checks from `backend/` with `uv run pytest -q`, `uv run ruff check .`,
`uv run ruff format --check .`, and `uv run basedpyright`. Build local derived data with
`uv run python -m app.pipeline.build`; install PyTorch with `uv sync --extra cpu` (or `cu128`)
and train with `uv run python -m app.model.train`; start the API with
`uv run uvicorn app.api.main:app`.
Keep the GRID data repository outside this project and set `STL_DATA_DIR` when it is not at the
documented sibling path. Never commit data or derived artifacts, including `model.pt` and
`model_metrics.json`.
