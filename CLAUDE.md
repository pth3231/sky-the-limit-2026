# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

**Always read [AGENTS.md](AGENTS.md) at the start of any session in this repository and follow its rules.** It contains binding instructions for AI agents (data handling, licensing, and workflow conventions including auto-closing issues on PR merge) that take precedence over the rest of this file.

## Status

The repository contains a Python backend in `backend/`: a streamed GRID data pipeline,
patch aggregates, a stateless FastAPI draft API, Fearless draft validation, and an explainable
statistical scorer. The PyTorch recommendation model, genetic algorithm, and Next.js frontend are
planned; see [README.md](README.md#roadmap) for the roadmap.

Run backend checks from `backend/` with `uv run pytest -q`, `uv run ruff check .`,
`uv run ruff format --check .`, and `uv run basedpyright`. Build local derived data with
`uv run python -m app.pipeline.build`; start the API with `uv run uvicorn app.api.main:app`.
Keep the GRID data repository outside this project and set `STL_DATA_DIR` when it is not at the
documented sibling path. Never commit data or derived artifacts.
