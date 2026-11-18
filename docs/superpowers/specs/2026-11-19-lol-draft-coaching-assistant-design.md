# LoL Draft Coaching Assistant — Design

Category 2 (LoL): Macro Scouting & Draft Management System — Cloud9 x JetBrains Sky's The Limit 2026.

## 0. Core principle

The primary user is a coach, not an engineer. They may have no interest in or
understanding of neural nets, genetic algorithms, or win-rate math. Every
user-facing output optimizes for **plain-language clarity first, statistical
accuracy second**: if a more precise explanation would require jargon or
hedging the coach can't act on, prefer the simpler framing. Confidence or
uncertainty, when shown, gets plain qualifiers ("strong signal" / "limited
data") instead of raw statistics (p-values, standard errors, feature
weights). This principle governs every explanation surfaced in the UI,
especially section 4's "contributing factors" layer.

## 1. Scope

An interactive League of Legends draft simulator for Cloud9 coaching
preparation:

- Cloud9 is always "our side"; the coach selects the opponent team.
- The coach configures the series format (Bo1 / Bo3 / Bo5) and a patch
  context before starting.
- The coach steps through the pick/ban phase. At each step the tool shows:
  the opponent's historical tendency for that slot, a recommended move for
  Cloud9 with a plain-language reason, a live predicted series win-rate, and
  a few runner-up alternatives.
- Fearless Draft carryover (any champion picked by either team in a prior
  game of the series becomes unavailable series-wide) is enforced
  automatically from Game 2 onward — this directly answers the brief's
  "how does the opponent's draft strategy shift in Game 3/4/5" question.
- Out of scope: ward/vision placement and jungle-pathing macro context. The
  GRID LoL event feed has no ward/vision events, only objective-timing
  events (drakes, Rift Herald, Baron, jungle camps, tower/plate
  destruction) — insufficient to support the brief's "warding tendencies"
  framing, so this panel is cut rather than built on data it can't
  actually represent.

## 2. Architecture

- **Backend:** Python, FastAPI. One language end-to-end — the data
  pipeline, NN training, NN live inference, GA, and the API all live in the
  same stack. No cross-language export/runtime-binding step.
- **Frontend:** React with TypeScript in strict mode. Renders the draft
  board, champion grid, live win-rate meter, and insights panel.
- **API shape:** stateless per request. The frontend POSTs the full current
  draft state (series format, game number, picks/bans so far) and receives
  `{ recommendation, winRate, alternatives, opponentTendencies,
  contributingFactors }`. This keeps the GA/model layer independently
  testable without server-side session state.
- **JetBrains tooling:** PyCharm for the backend, WebStorm for the
  frontend.

## 3. Data pipeline

A build-time job (not part of the live request path) that:

1. Reads `meta/games.csv`, `meta/characters.csv`, `meta/teams.csv` from the
   GRID data repo (referenced by path via `STL_DATA_DIR`, never committed —
   see `AGENTS.md`).
2. For each series' `lol/events_{seriesId}_grid.jsonl.zst` feed, reads the
   last envelope with a full `seriesState` snapshot and extracts
   `game.draftActions` for each game in the series — this is the
   structured pick/ban record already present in the snapshot, so there is
   no need to replay every `team-picked-character` /
   `team-banned-character` event by hand.
3. Produces patch-bucketed aggregates: per-champion win/pick/ban rate, blue
   vs red side win-rate delta, champion-pair synergy (co-pick win rate),
   and per-team (Cloud9 and each opponent) historical performance on/against
   each champion.
4. Writes these aggregates to disk (e.g. Parquet/SQLite, already covered by
   `.gitignore`) for the live API to read — the live path never re-parses
   raw event feeds per request.

**Patch recency rule:** prefer the most recent patch present in the
dataset for a given opponent/champion; if there are too few games on that
patch, blend in the next most recent patches rather than relying on a
handful of games. Note: this dataset's LoL coverage ends at LCS Promotion
2026 (2026-10-12), so "most recent patch in the dataset" may lag the
actual current patch by several weeks — this is a property of the fixed
hackathon snapshot, not a pipeline bug.

**Fearless carryover source:** `meta/games.csv` already excludes
rolled-back/abandoned games (it's built from each feed's final state), so
it can be used directly for game-level results and patch info without
re-implementing GRID's rollback-handling caveats. Draft pool exclusions for
Fearless Draft are tracked separately per simulated series (see section 5),
not derived from `meta/games.csv`.

## 4. Recommendation engine (GA + NN)

- **Fitness function (live):** a neural net, trained offline (PyTorch) on
  historical draft-state → win-probability examples built from the
  pipeline's aggregates. This NN is the actual score the GA optimizes for
  accuracy.
- **GA (next-move-only, re-run live on every pick/ban):** for the current
  slot, the population is the set of still-legal candidate champions
  (respecting Fearless carryover for the current simulated series). Fitness
  for each candidate = the NN's predicted win probability assuming the
  opponent's most likely reply, drawn from their tendency distribution for
  that slot. Scoped to the next move only (not a deeper lookahead) to keep
  the per-click search space small and the response fast.
- **Explanation layer ("contributing factors"):** a separate, explainable
  statistical blend — champion win rate, side delta, synergy, Cloud9's and
  the opponent's history with the champion — is computed per candidate and
  shown alongside the NN's recommendation. This is explicitly an
  **approximation** of the NN's reasoning, not literally what drove its
  number, and the UI must not imply otherwise. Per the core principle
  (section 0), this layer is phrased in coaching language ("You tend to
  win more on blue side with this pick," "They've answered this ban with
  Sejuani 70% of the time"), never in ML terms ("side_delta: +0.23").

## 5. Fearless Draft / series flow

The coach configures the opponent, series format, and starting patch
context. Game 1 drafts against the full legal champion pool. After each
game ends, every champion picked by either team (not bans) joins a
series-wide excluded pool for the rest of the simulated series. Games 2–5
re-run the recommendation engine against the shrinking legal pool —
producing visibly different recommendations as the series progresses,
which is the concrete mechanism behind the brief's "how does draft
strategy shift by Game 3/5" question.

## 6. UI/UX

- **Draft board:** champion grid plus ordered pick/ban slots in standard
  draft order, with Fearless-excluded champions visibly greyed out, and a
  series tracker showing games completed so far.
- **Live win-rate meter:** updates after every locked-in move.
- **Insights panel:** the top NN-scored recommendation, its
  contributing-factors breakdown (plain language per section 0), a few
  runner-up alternatives with their scores, and the opponent's tendency
  stats for the current slot. This panel also serves as the
  previously-separate "monitoring dashboard" idea — folded in as one
  coach-facing surface rather than a separate internal tool.
- **Meta badge (stretch goal, section 8):** if the external high-elo
  pipeline is built, a small separate badge per champion shows current
  Diamond+ tier/win rate, clearly labeled as a distinct signal from the
  pro-play tendency stats it sits next to.

## 7. Testing

- **Statistical-blend factors:** unit tests against known champion/patch
  stat fixtures, asserting expected ordering of contributing factors.
- **Fearless carryover:** unit tests asserting the legal pool shrinks
  correctly game-to-game and never re-admits an excluded champion within a
  series.
- **NN:** a held-out validation split evaluated for basic accuracy/AUC
  against historical game outcomes, so there's a concrete number to cite
  in the submission writeup.
- **GA legality:** tests asserting the GA never recommends a champion that
  is already picked, already banned, or Fearless-excluded for the current
  series state.

## 8. External data (stretch goal, not core)

Ship the GRID-only version first, fully working end-to-end. If time
remains, optionally add:

- **Riot Developer API** (`league-v4`/`league-exp-v4` + `match-v5`) for
  Diamond–Challenger solo-queue champion stats, including ban data
  present in ranked match payloads. Caveat: personal API keys are
  rate-limited and expire every 24 hours unless a persistent key is
  granted by Riot, which is an application process, not instant — plan
  around re-issuing the key if this is pursued.
- **Leaguepedia** (community wiki, CC-BY-SA) for pro-match coverage past
  this dataset's October 2026 cutoff, if "most recent competitive league
  data" is prioritized over solo-queue stats.
- Explicitly avoided: scraping community tier-list sites (U.GG, OP.GG,
  Mobalytics, Lolalytics) without confirmed permission — likely a Terms of
  Service violation, not something to build into a hackathon submission.

Either source surfaces only as the meta badge in section 6, and never
blocks or changes the core demo if it isn't finished in time.
