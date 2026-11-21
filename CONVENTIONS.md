# Conventions

Solo project. `main` is protected — all changes land via PR.

## Commits

[Conventional Commits](https://www.conventionalcommits.org/): `<type>: <description>`

Types: `feat`, `fix`, `docs`, `chore`, `refactor`, `test`, `style`, `perf`

Examples:
- `feat: add Fearless Draft carryover logic`
- `fix: GA recommends already-banned champion`
- `docs: update README with draft tool scope`

## Branches

`<type>/<issue-number>-<slug>`, using the same types as commits.

Examples: `feat/12-draft-board`, `fix/7-ga-legal-pool`

Every branch traces back to an issue — see Issues below.

## Issues

- Title: `[Area] Short description` (e.g. `[GA] Legal-pool check fails after Game 2`).
- One type label per issue: `bug`, `feature`, `chore`, `question`.
- **Every task requires an issue.** Open an issue before starting any work —
  code, documentation, or conventions. The branch name's issue number depends
  on it, and the PR must reference it. No exceptions, not for small fixes or
  documentation-only changes.
- **Named rule — The Issue-First Rule:** An issue exists before a branch is
  created. If you realize mid-work that no issue was opened, stop and create
  one before committing. This ensures every change is traceable and every PR
  has an issue to close.

## Pull requests

- One PR per issue.
- PR title: Conventional Commits format, same as commit messages — it
  becomes the final commit message on `main` via squash merge.
- **Auto-close the related issue:** The PR description **must** include
  `Closes #<issue-number>` on its own line so GitHub automatically closes
  the issue when the PR is merged. Additionally, add a `Closes: #<N>`
  trailer to the commit message (e.g. `git commit --trailer "Closes: #9"`)
  so the link is preserved in history even if the PR body is lost.
- Merge strategy: **squash merge only**. Keeps `main` linear, one commit
  per PR.
- No required reviewer (solo project) — merge once the PR is ready and any
  checks pass.
- **Before merging:** Verify the `Closes #<N>` keyword is present in the PR
  body. If the PR addresses multiple issues, list each one (e.g.
  `Closes #9, #10`).

## What gets committed

For this project, IDE config and docs are part of the deliverable and are
committed **alongside** the code they relate to (same PR), not in separate
cleanup PRs:

- **`.idea/`** — shared PyCharm/WebStorm project config (`*.iml`,
  `modules.xml`, `vcs.xml`, `misc.xml`, `inspectionProfiles/`, run
  configurations). Committed as hackathon evidence of JetBrains tooling use.
  Personal/local files stay out via `.idea/.gitignore` (`workspace.xml`,
  `shelf/`, `dataSources/`, `dataSources.local.xml`, `httpRequests/`,
  `queries/`). Never commit a data source or run configuration that embeds
  GRID data, credentials, or absolute paths into the data repo.
- **`.junie/`, `.air/`, `.run/`** — Junie/Air plan and run-configuration
  artifacts, same rationale as `.idea/`: Cloud9 confirmed in the hackathon
  Discord that committing these as JetBrains-tooling evidence is encouraged.
  Check for secrets or absolute local paths before committing, same as
  `.idea/`.
- **`docs/`** — specs (`docs/superpowers/specs/`), implementation plans
  (`docs/superpowers/plans/`), and any other design notes. When a change
  alters a decision, update the relevant spec/plan/`README.md`/`PRODUCT.md`
  in the same PR.
- **Top-level context files** — `AGENTS.md`, `CLAUDE.md`, `PRODUCT.md`,
  `DESIGN.md`, `CONVENTIONS.md`, `README.md`.

**Never commit or push anything derived from the GRID data** (see
`AGENTS.md`): no `stl-hackathon-2026-grid-starter/` contents, `.zst` /
`.jsonl` files, `meta/` or `series.csv` copies, extracts, caches, Parquet /
SQLite / DuckDB builds, trained model weights fitted on the data, or test
fixtures copied from real feeds (hand-write synthetic fixtures instead).
Derived artifacts go to an ignored directory. Before every commit:

- Stage files explicitly (`git add <path>`), never `git add -A` / `git add .`.
- Check `git diff --cached --name-only` and confirm no data-derived file is
  staged.
- If data was ever committed, stop — do not push — and fix history first.

## Hackathon submission compliance

Rules confirmed by Cloud9/Devpost staff in the hackathon Discord
(2026-11-18 through 2026-11-21), kept here since they affect how this
project is built and demoed, not just the data handling already covered
above and in `AGENTS.md`:

- **No single required AI tool.** The JetBrains-tool requirement is
  satisfied by using any JetBrains IDE (PyCharm, WebStorm, IntelliJ IDEA,
  Rider, DataGrip), JetBrains Air, or Junie — not Junie specifically. This
  project's primary tools are PyCharm and WebStorm.
- **Declare JetBrains tool use in the final submission** — state how and
  where each tool was used; Cloud9 verifies this. Committing `.idea/`,
  `.junie/`, `.air/`, and `.run/` (see "What gets committed" above) is
  encouraged supporting evidence.
- **A hosted public demo may serve real statistics/insights derived from
  the GRID data** — this is the intended use of the data grant, not a
  violation. A password-protected demo visible only to judges is
  acceptable. The app must not require a local file upload to function.
- **Any fabricated/synthetic data shown in a demo must be clearly labeled
  as such** to the viewer (e.g. this project's frontend "practice mode"
  already labels itself visibly). Don't substitute synthetic data for real
  data that was available — that's discouraged, not prohibited, but judges
  may question it.
- **Riot Developer API use is allowed alongside GRID data** — see
  `AGENTS.md` for the key-security and API-approval requirements before
  adding any Riot API integration.

## Coding conventions

Shared across both stacks:

- **Test-driven development is mandatory**: write the failing test first,
  then the minimal implementation, then refactor. No production code lands
  without a test that exercised it red → green.
- Max line length: **100** characters (Ruff and Prettier both configured to
  this).
- Docstrings/doc comments: standard style — Python uses PEP 257 docstrings
  (summary line, blank line, detail); TypeScript uses JSDoc/TSDoc on
  exported symbols. Match the existing comment density of the file.

### Python (backend: FastAPI + PyTorch)

- **Toolchain:** `uv` with a single `backend/pyproject.toml` (+ `uv.lock`);
  no `requirements.txt`.
- **Lint + format:** Ruff (lint and formatter), configured in
  `pyproject.toml`, `line-length = 100`.
- **Type checking:** Pyright (`basedpyright`), `standard` mode. Annotate all
  public functions; untyped third-party stubs are tolerated.
- **Layout:** flat `backend/app/` package (e.g. `app/pipeline/`,
  `app/engine/`, `app/api/`), no `src/` indirection.
- **Tests:** `backend/tests/` mirroring `app/` — `tests/test_<module>.py`,
  pytest, shared fixtures in `conftest.py`.
- **Data models:** Pydantic v2 everywhere — API schemas and internal
  pipeline/engine state alike.
- **Naming:** `snake_case` in Python; API models use an alias generator so
  the wire format is `camelCase` (e.g. `win_rate` ↔ `winRate`).

### TypeScript (frontend: React)

- **Framework:** Next.js with TypeScript in **strict mode**.
- **Lint + format:** ESLint (`typescript-eslint`, `react-hooks`) + Prettier,
  `printWidth: 100`.
- **Styling:** Tailwind CSS; design tokens from `DESIGN.md` are mapped into
  the Tailwind config, not hard-coded in components.
- **State:** Zustand store for draft state.
- **Components:** PascalCase folder-per-component with named exports —
  `components/DraftBoard/DraftBoard.tsx`, `DraftBoard.test.tsx`.
- **Tests:** Vitest + React Testing Library for unit/component tests;
  Playwright for E2E. Playwright runs against a production build
  (`next build && next start`), not the lazy-compiling dev server, so
  timing and memory behaviour match deployment.
- **API types:** generated from FastAPI's `/openapi.json` via
  `openapi-typescript`; Pydantic models are the single source of truth,
  never hand-write wire types. Wire JSON is `camelCase`.
