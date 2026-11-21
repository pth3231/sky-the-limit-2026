# AGENTS.md

Instructions for AI coding agents and assistants working with the **Sky's The Limit 2026 – Cloud9 x JetBrains Hackathon** GRID data. Human-oriented documentation for the data itself (event format, recipes, metadata tables, data caveats) is in the data repo's own `README.md` (`$STL_DATA_DIR/README.md`, default `../stl-hackathon-2026-grid-starter/README.md` — not this project repo's `README.md`, which is a different file; see below). The data license is that repo's `LICENSE.md`.

## Context files to read first

Before doing any work in this repo, read these in order. Each owns a distinct kind of context — don't duplicate one file's content into another.

1. **`CLAUDE.md`** — harness-level instructions for Claude Code; points back to this file as binding.
2. **`AGENTS.md`** (this file) — GRID data handling and licensing rules (below). Binding for any agent, regardless of harness.
3. **`PRODUCT.md`** — durable product truth: users, purpose, positioning, capabilities/constraints, product principles. Does not contain visual design.
4. **`DESIGN.md`** — the visual design system (currently a seed, pre-implementation; colors and fonts are marked unresolved until real code exists to extract them from).
5. **`CONVENTIONS.md`** — commit message, branch, issue, and PR naming/workflow conventions.
6. **`docs/superpowers/specs/*.md`** — brainstormed design specs for specific features (currently: `2026-11-19-lol-draft-coaching-assistant-design.md`, the LoL draft coaching assistant design).
7. **This project's own `README.md`** — project overview and deployment notes. Distinct from the GRID data repo's `README.md` referenced above.

## Rules and facts

- **🚫 Never commit the data (the most important rule).** Don't add any file from this repository to the user's project repository. That covers `.zst` files, decompressed `.jsonl` files, `meta/` or `series.csv` copies, and any extract, cache or database built from the data. Committing and pushing it is redistribution under the [LICENSE](LICENSE.md), even in a private repo. When you set up a project:
  - Keep this data repository outside the project directory and reference it by path, for example `STL_DATA_DIR`, defaulting to `../stl-hackathon-2026-grid-starter`.
  - Add the `.gitignore` entries from [Keep the data out of your project repo](README.md#keep-the-data-out-of-your-project-repo).
  - Write derived data to an ignored directory.
  - Before every commit, check the staged files (`git diff --cached --name-only`). Don't stage everything blindly with `git add -A`/`git add .` if data files might be present.
  - If data was committed, stop and tell the user rather than pushing.

- **Data contract.** Files are `{game}/events_{seriesId}_grid.jsonl.zst` (one GRID series each) and `val-ability-supplements/{seriesId}.jsonl.zst`. Format: zstd-compressed UTF-8 JSON Lines, standard frame, no dictionary, default decoder settings. One envelope per line, ordered by `sequenceNumber`. Envelope keys: `id, correlationId, occurredAt, seriesId, sequenceNumber, events[]`. Event keys: `id, type, actor, action, target, seriesState, seriesStateDelta, includesFullState`. Invariant: `type == f"{actor.type}-{action}-{target.type}"`.
- **Index first.** Read `val/series.csv` / `lol/series.csv` to choose series, and `meta/*.csv` for teams, players, tournaments, characters, per-game results and rosters. Many questions (who played which agent or champion, map results, head-to-heads) can be answered from `meta/` alone, without opening a feed. Don't list or glob directories to discover matches, and don't open every file to find one. IDs in `meta/` are unique per `title` (`val`/`lol`).
- **Prefer streaming in code; decompress files when someone wants to read them.** Uncompressed data is 50–150× the compressed size (about 160 GB for VALORANT, up to about 1 GB per match). In code, iterate line by line with `compression.zstd.open(path, "rt")` (Python 3.14+) or `zstandard.open` (`pip install zstandard`) rather than loading whole files into memory. Decompressing individual files to plain `.jsonl` (for example `zstd -d file.jsonl.zst`) is fine whenever the user wants to inspect them or a tool needs plain text. Check with the user before decompressing many files or the whole corpus, because of the disk space involved. Keep any decompressed copies out of git (see the first rule).
- **Inspect cheaply.** To learn the schema, read the first few envelopes and the last one of a single file. Don't print whole envelopes, because `seriesState` is a full snapshot and can be large.
- **Final results** come from the `seriesState` of the last event that has one. Per-event changes are in `seriesStateDelta`.
- **Joins.** Supplement `grid_event_id` = GRID `events[].id`. Supplement `occurred_at` = envelope `occurredAt`. `player_id`/`game_id` are GRID IDs. All IDs are strings.
- **Other license constraints.** Don't upload, publish or mirror the data, or derived datasets containing it, anywhere outside the user's machine. Don't add the data to model-training pipelines that outlive the hackathon. The license to use the data ends when the hackathon ends (copies don't have to be deleted unless Cloud9 asks in writing, but must not be used). Projects should read data from a local clone of this repository. Small excerpts are acceptable only to demonstrate the project to judges. See [LICENSE](LICENSE.md).

## Workflow conventions for AI agents

- **Issue-first rule (The Issue-First Rule):** Every task requires a GitHub
  issue before any work begins — code, documentation, or convention changes.
  Create the issue before branching. If you realize mid-work that no issue
  exists, stop and create one before committing. This ensures every PR has
  an issue to close and every change is traceable. See
  [CONVENTIONS.md](CONVENTIONS.md#issues) for details.
- **Auto-close issues on PR merge:** When creating a Pull Request, the PR
  description **must** include `Closes #<issue-number>` on its own line so
  GitHub automatically closes the related issue when the PR is merged. If
  the PR addresses multiple issues, list all of them (e.g. `Closes #9, #10`).
  Additionally, add a `Closes: #<N>` trailer to the final commit message
  using `git commit --trailer "Closes: #<N>"` so the link survives squash
  merges. See [CONVENTIONS.md](CONVENTIONS.md#pull-requests) for the full
  PR workflow.
- **Do not leave issues open after merging:** If a PR has been merged and the
  related issue is still open, close it manually with a comment referencing
  the merged PR.