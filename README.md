# Draft Coach

Project for the **Cloud9 x JetBrains: Sky's The Limit Hackathon 2026** —
Category 2 (LoL): Macro Scouting & Draft Management System.

# What am I building?

**Draft Coach**: an interactive League of Legends draft simulator for Cloud9
coaching prep. Full product truth lives in [PRODUCT.md](PRODUCT.md); full design rationale
in [docs/superpowers/specs/2026-11-19-lol-draft-coaching-assistant-design.md](docs/superpowers/specs/2026-11-19-lol-draft-coaching-assistant-design.md).
This section is a summary, not the source of truth.

Cloud9 is always "our side." The coach picks an opponent and a series
format (Bo1/Bo3/Bo5), then steps through the pick/ban phase. At each slot:

- **A recommended move with a plain-language reason** — e.g. "They've
  picked champion A 70% of the time in Game 1. Banning it now costs their
  Fearless Draft win rate n% in Game 2."
- **A live predicted series win-rate**, updating after every locked-in move.
- **A few runner-up alternatives**, so the coach sees what else the engine
  considered.
- **Fearless Draft carryover**, enforced automatically from Game 2 onward:
  any champion picked by either team in a prior game is excluded for the
  rest of the series, and recommendations visibly shift as the pool shrinks.

Recommendation engine: a neural net, trained offline on historical
draft-state → win-probability data, is the live fitness function a genetic
algorithm uses to score the next move (next-move-only, re-run on every
pick/ban). A separate, explainable statistical blend (champion win rate,
side delta, synergy, team history) is computed alongside it purely as the
human-readable "why" — an approximation of the NN's reasoning, not
literally it. **Core principle:** every output optimizes for plain-language
clarity first, statistical accuracy second — the primary user is a coach,
not an engineer.

Stack: React + TypeScript (strict) frontend, Python/FastAPI backend
(data pipeline, NN training/inference, GA, API) — see
[PRODUCT.md](PRODUCT.md#stack). Out of scope: ward/vision or jungle-pathing
macro context (the GRID LoL feed has no ward events). Stretch goal only,
never blocking the core demo: Riot Developer API (Diamond–Challenger solo
queue) or Leaguepedia data, shown as a separate "current meta" badge.

# Repository map

| Path | What it is |
| --- | --- |
| [PRODUCT.md](PRODUCT.md) | Durable product truth: users, purpose, positioning, capabilities, principles. |
| [DESIGN.md](DESIGN.md) | Visual design system (seed; direction: "Broadcast Sequence Bar"). Colors/fonts resolved during implementation. |
| [docs/superpowers/specs/](docs/superpowers/specs/) | Approved design spec(s): architecture, data pipeline, GA+NN engine, Fearless flow, UI, testing. |
| [CONVENTIONS.md](CONVENTIONS.md) | Commit, branch, issue, PR, and coding conventions (Python + TypeScript). |
| [AGENTS.md](AGENTS.md) / [CLAUDE.md](CLAUDE.md) | Rules for AI coding agents — most importantly, GRID data handling. |
| `backend/` | Python / FastAPI (`uv`, Ruff, Pyright, Pydantic v2): data pipeline, draft API, PyTorch NN training + inference, and GA search (built). |
| `frontend/` | Next.js + TypeScript strict (Tailwind, Zustand, Vitest): draft board, champion grid, series controls, and insights. |

# Working with the data

The GRID esports dataset (`stl-hackathon-2026-grid-starter`) is licensed for
the hackathon only and is **never committed** to this repo. Keep its clone
outside this project and point the project at it via `STL_DATA_DIR` (default:
the sibling `../stl-hackathon-2026-grid-starter`). `.gitignore` excludes the data
directory, `*.zst`/`*.jsonl` files, and derived artifacts
(`*.parquet`, `*.sqlite`, `*.duckdb`). Before every commit, check the
staged files (`git diff --cached --name-only`).

How the pipeline uses it (details in the spec, section 3):

- `meta/games.csv`, `meta/characters.csv`, `meta/teams.csv` for results,
  patch, side, and ID lookups. `meta/games.csv` already excludes
  rolled-back games. Cloud9's LCS entry is "Cloud9 Kia" (`team_id 47351`).
- Per-series `lol/events_{seriesId}_grid.jsonl.zst` feeds, read once at
  build time: the final `seriesState` snapshot carries `game.draftActions`
  (structured pick/ban record), so no event replay is needed.
- Aggregates are patch-bucketed and written to an ignored directory; the
  live API never re-parses raw feeds. LoL coverage ends at LCS Promotion
  2026 (2026-10-12), so the newest patch in the data may lag the live
  patch — the "most recent patch, blend in older if thin" rule covers this.

# Running the backend

From `backend/` (Python 3.12 via `uv`):

```bash
cp .env.example .env                    # first-time setup; then adjust local paths if needed
uv sync --extra cpu                          # install dependencies (or --extra cu128 for CUDA)
uv run python -m app.pipeline.build          # GRID data -> backend/.derived/ (ignored)
uv run --extra cpu python -m app.model.train  # train model.pt + model_metrics.json (ignored)
uv run --extra cpu uvicorn app.api.main:app --reload  # API on http://127.0.0.1:8000
uv run pytest -q && uv run ruff check . && uv run basedpyright
```

Use `--extra cu128` instead of `--extra cpu` in the model commands for CUDA. Backend
settings load from `backend/.env`; process environment variables override values in that
file. Copy `backend/.env.example` once and adjust `STL_DATA_DIR` if the GRID clone is not
beside the project. Keep the separately licensed data clone outside this project; do not
copy its files here. Relative backend paths are resolved from the working directory, so
run the commands above from `backend/`.

The build also tries to cache Data Dragon champion traits (`ddragon.json`); it never fails
without network, and `--no-ddragon` skips it. Training splits chronologically by series (newest
15% held out); training examples use deterministic series-grouped out-of-fold history, while
stopping and held-out examples use features fitted on the earlier training window. It reports
accuracy, AUC and log loss against a blue-side baseline and the statistical scorer;
`--device auto|cpu|cuda` and `--seed` are available. PyTorch is optional:
when `model.pt` is missing or PyTorch isn't installed, the API falls back to the statistical
scorer, and every response reports which one ran in its `engine` field (`model` or
`statistical`).

Settings come from `backend/.env` or the process environment: `STL_DATA_DIR` (data repo),
`DRAFT_COACH_DERIVED_DIR` (default `backend/.derived`; also used as the
pipeline build output unless `--out` is passed), and `CLOUD9_TEAM_ID`
(default: the LoL team whose name starts with "Cloud9"). Endpoints:
`GET /health`, `/teams`, `/champions`, `/patches`, and `POST /draft/recommend`,
which takes the full series and draft state (camelCase JSON) and returns the
recommended move, alternatives, game and series win rates, opponent
tendencies, and plain-language contributing factors. Illegal draft states
(Fearless repeats, duplicates, unknown champions or patches) return 422.

# Running the frontend

From `frontend/` (Node.js 20.9 or newer):

```bash
cp .env.example .env.local           # first-time setup; Next.js loads this automatically
npm ci
npm run dev                 # http://localhost:3000
npm test                    # unit and component tests
npm run lint && npm run typecheck && npm run build
```

Next.js loads `frontend/.env.local` automatically. The frontend proxies `/api/*` to
`http://127.0.0.1:8000` by default; change `DRAFT_COACH_API_URL` there to use a different
backend address. Start the backend with local aggregates as described above
to load actual opponents, champions, patches, and recommendations. The response's
`engine` field identifies whether the neural model or statistical scorer produced the
estimates; the workspace labels those methods separately. Statistical factors and
opponent tendencies provide historical context and are not model predictions.

For a production-mode local check, run `npm run build` followed by `npm run start`.
The proxy address is baked in at build time: if the backend port changes, set
`DRAFT_COACH_API_URL` and rebuild. Check that the selected port runs this version of
the backend rather than a stale server.

If the backend is unavailable, choose **Practice mode** to explore the draft workflow
with explicitly synthetic champions and no recommendations or win-rate claims. Drafts
live in the browser session only. Changing a series setting resets the draft and previous
games; a completed game's result must be recorded before the next game starts. Only
picks carry forward as Fearless exclusions, and a decided series cannot continue.

API types in `frontend/lib/api-types.ts` are generated from the backend's OpenAPI
schema. With the API running on the default port, regenerate them from `frontend/` with
`npm run generate:api` after changing the backend contract.

# Development workflow

Solo project; `main` is protected. Every change lands via a PR that closes
an issue, squash-merged with the PR title as the commit message. Branches
are `type/issue-number-slug`; commits follow Conventional Commits. Full
details in [CONVENTIONS.md](CONVENTIONS.md).

Tooling: PyCharm (backend) and WebStorm (frontend). `.idea/` (shared
project config) and `docs/` (specs, plans) are committed alongside the code
they relate to, in the same PR; nothing derived from the GRID data is ever
committed or pushed — see
[CONVENTIONS.md](CONVENTIONS.md#what-gets-committed). Test-driven development is mandatory;
linters, formatters, and type checkers per stack are listed in
[CONVENTIONS.md](CONVENTIONS.md#coding-conventions).

# Roadmap

Implementation plans are written per subsystem (`docs/superpowers/plans/`)
in this order:

1. **Data pipeline** — `meta/*.csv` + GRID feeds → patch-bucketed
   aggregates + `draftActions` extraction.
2. **Backend core** — FastAPI API, Fearless draft state machine,
   statistical-blend "why" scorer.
3. **Recommendation engine** — PyTorch NN fitness function + rest-of-phase GA (built; see
   `docs/superpowers/specs/2026-11-19-recommendation-engine-design.md`).
4. **Frontend draft workspace** — Next.js + TS strict, implementing the
   DESIGN.md direction; TS API types generated from FastAPI's OpenAPI.

# License

Project code and docs are released under the [MIT License](LICENSE)
(hackathon OSS submission requirement). The GRID dataset is **not** covered
by it — it stays under its own proprietary license in the data repo
(`$STL_DATA_DIR/LICENSE.md`).

# Deployment notes (free hosting, for later)

Need: a public, judge-accessible link (hackathon only requires "a functional
access link," not a custom domain) plus the code repo.

**Frontend (React/TS build):** Vercel, Netlify, or Cloudflare Pages — all
free, auto HTTPS, custom domain support, no card required.

**Backend (FastAPI + NN inference):**
- **Hugging Face Spaces (Docker SDK)** — free CPU tier, no card, built for
  ML demos. Likely pick, given the NN is the centerpiece.
- **Render free web service** — free, no card, but spins down after ~15 min
  idle (30-50s cold start on next hit). Fine for occasional judge clicks,
  risky for a live stage demo.
- **Oracle Cloud "Always Free" VM** — free forever, no spin-down, full
  control, but needs a card on file (never charged on free tier) and more
  manual setup (nginx, systemd).

**Leaning:** Vercel (frontend) + Hugging Face Spaces (backend) — fastest
to stand up, no card requirement, no cold-start risk during a live demo.

**Domain:** a free subdomain (`*.vercel.app`, `*.hf.space`) already
satisfies the hackathon's submission requirement — no registration needed.
If a custom root domain is still wanted: GitHub Student Developer Pack
gives a free `.me` domain via Namecheap for a year (if eligible as a
student). Avoid Freenom (`.tk`/`.ml`/`.ga`/`.cf`/`.gq`) — unreliable since
Meta's lawsuit against them, domains get seized unpredictably.
