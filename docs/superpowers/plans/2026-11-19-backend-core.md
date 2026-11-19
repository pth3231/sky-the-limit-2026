# Backend Core — Implementation Plan

**Issue:** #6 — `[Feature] Add backend draft core and recommendation API`
**Roadmap step:** 2 — backend core (draft rules, explainable scorer, API)

## Goal

A stateless FastAPI backend that takes the full current draft state, enforces the standard pro
draft order and Fearless rules, and returns a recommended move with plain-language reasons,
runner-up alternatives, the live game and series win rate, and the opponent's tendencies. The
ranking is an explainable statistical blend over the step-1 aggregates. It stands in until the
GA + NN of step 3, and stays as the "contributing factors" explanation layer afterwards.

## Decisions

- **Stateless API.** The client sends the series format, our side, the opponent, the optional
  patch, every previous game's picks and result, and the current game's actions (champion IDs in
  draft order). The server keeps no session state.
- **Draft order.** The standard 20-slot pro order: bans B R B R B R, picks B R R B B R, bans
  R B R B, picks R B B R. Each side gets 5 bans and 5 picks.
- **Fearless.** A champion picked by either team in an earlier game of the series is unavailable
  for the rest of the series, for picks and bans alike. Bans do not carry over.
- **Patch blending.** Start from the requested patch (default: newest). Add older patches, newest
  first, until at least 60 games are covered. Newer patches than the requested one are never
  used. An unknown patch is a 422.
- **Scoring.** Smoothed rates with a 50% prior (pseudo-count 10) so small samples stay near
  even. A team's strength is the sum of per-champion meta edges (capped ±10 points), team comfort
  edges (team win rate with the champion minus its overall win rate, capped ±5) and same-team
  pair synergy edges (pair win rate minus the pair's average, capped ±5). Game win probability
  is `sigmoid(1.2 × (side edge + our strength − their strength))`, about `0.5 + 0.3x` near even.
  The signals overlap, so the gap is damped: a first real-data smoke test with slope 4 rated
  finished drafts at 97–98%, which no coach would trust; slope 1.2 brings them to 74–82%.
- **Candidate scores.** Picks: the game win probability after adding the champion. Bans: the
  current probability plus the protected margin (how much the opponent would gain with it),
  weighted by how often the opponent picks it.
- **Series win rate.** Exact best-of-N recursion from the current score, using the current game's
  probability for this game and 50% for later games.
- **Plain language.** Factors read like "Wins 56% of pro games on recent patches (40 games)" or
  "Strong with your Lee Sin (won 7 of 9 together)". Each has a signal qualifier: strong (30+
  games), some (10+) or limited. No ML terms.
- **Our team.** `CLOUD9_TEAM_ID` if set; otherwise the LoL team whose name starts with "Cloud9".
- **Wire format.** camelCase JSON through a Pydantic alias generator; snake_case in Python.

## Scope and touch points

- `backend/pyproject.toml`, `backend/uv.lock`: add `fastapi`, `uvicorn`; dev `httpx2` (the
  Starlette test client deprecates `httpx`).
- `backend/app/config.py`: `derived_dir` (`DRAFT_COACH_DERIVED_DIR`, default `backend/.derived`)
  and `cloud9_team_id` (`CLOUD9_TEAM_ID`) settings.
- `backend/app/pipeline/aggregates.py`: make `patch_sort_key` public for reuse, and add per-team
  `teams` totals (games, decided games, wins) so opponent tendency rates use exact game counts.
- `backend/app/engine/errors.py`: `DraftStateError`, `UnknownPatchError`.
- `backend/app/engine/models.py`: camelCase request/response models.
- `backend/app/engine/draft_order.py`: the 20-slot order and slot ownership.
- `backend/app/engine/fearless.py`: Fearless exclusions and draft-state validation.
- `backend/app/engine/stats.py`: patch blending into summed lookups.
- `backend/app/engine/scoring.py`: smoothed rates, team strength, factors, series math.
- `backend/app/engine/service.py`: `DraftEngine.recommend` and `load_engine`.
- `backend/app/api/main.py`: `create_app`; `GET /health`, `/teams`, `/champions`, `/patches`;
  `POST /draft/recommend`; draft-state errors map to 422.
- `README.md`: how to build the derived data and run the API.

Out of scope: NN/GA (step 3), frontend (step 4), deployment.

## Tasks (test first, red → green, per task)

1. **Draft order** — 20 slots; 5 bans and 5 picks per side; exact sequence; red-side ownership.
2. **Fearless and validation** — pool shrinks game 1 → 2 → 3; bans don't carry over; rejects
   excluded, duplicate or unknown champions, shared picks across games, more than 5 picks per
   side, more than 20 actions, and a series that is already decided.
3. **Patch blending** — numeric order (`16.10` newer than `16.9`); stops at the threshold; an
   older requested patch excludes newer ones; counts are summed; unknown patch raises.
4. **Scoring** — a strong champion outranks a weak one; the synergy factor names the teammate;
   limited-data qualifier; factors ordered by impact; series math (Bo1 = p; Bo3 at 1–0 with
   p = 0.5 → 0.75).
5. **Service** — recommendation and alternatives are never picked, banned or excluded; the
   opponent's slot returns tendencies and no recommendation; a complete draft still reports win
   rates; unknown opponent raises.
6. **API** — camelCase in and out; 422 on illegal state; list endpoints.
7. **Smoke** — run the engine on the real local aggregates and check shape and legality only.
   Nothing from the data is printed into docs or committed.

All fixtures are synthetic, built in the tests from hand-written `GameDraft` records.
