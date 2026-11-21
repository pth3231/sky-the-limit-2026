# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

**Always read [AGENTS.md](AGENTS.md) at the start of any session in this repository and follow its rules.** It contains binding instructions for AI agents (data handling, licensing, and workflow conventions including auto-closing issues on PR merge) that take precedence over the rest of this file.

## Status

The repository contains a fully implemented draft coaching tool:
- **Backend** (`backend/`): streamed GRID data pipeline, patch aggregates, stateless FastAPI
  draft API, Fearless draft validation, explainable statistical scorer, PyTorch recommendation
  model with GPU acceleration, and genetic algorithm for chain recommendations.
- **Frontend** (`frontend/`): Next.js + TypeScript strict app with Tailwind CSS, Zustand state
  management, champion grid with portraits, stats panel with win-rate chart, and a draft bar
  showing both teams' bans and picks.

Run backend checks from `backend/` with `uv run pytest -q`, `uv run ruff check .`,
`uv run ruff format --check .`, and `uv run basedpyright`. Run frontend checks from
`frontend/` with `npx vitest run` and `npx tsc --noEmit`. Build local derived data with
`uv run python -m app.pipeline.build`; start the API with `uv run uvicorn app.api.main:app`.
Start the frontend with `cd frontend && npm run dev`.
Keep the GRID data repository outside this project and set `STL_DATA_DIR` when it is not at the
documented sibling path. Never commit data or derived artifacts.
