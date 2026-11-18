# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Stack

React with TypeScript (strict mode) frontend; Python/FastAPI backend (data
pipeline, NN training and live inference, GA, API). Decided during design,
not delegated — see `docs/superpowers/specs/2026-11-19-lol-draft-coaching-assistant-design.md`.

## Users

Cloud9's League of Legends coaching staff, preparing for an upcoming
opponent match. Primary job: step through pick/ban phase prep (and
Fearless Draft series scenarios across Games 1-5) and get an
opponent-specific, win-rate-grounded recommendation for the next move,
without needing to understand the modeling underneath it. The primary
user may have no technical background — see Product Principles.

## Product Purpose

"Draft Coach" is an interactive League of Legends draft simulator for
pre-match coaching prep. Cloud9 is always "our side." The coach selects an
opponent, a series format (Bo1/Bo3/Bo5), and a patch context, then steps
through the draft. At each step the tool surfaces the opponent's
historical tendency for that slot, a recommended move with a plain-
language reason, a live predicted series win-rate, and a few runner-up
alternatives. Built for Category 2 (LoL) of the Cloud9 x JetBrains Sky's
The Limit 2026 hackathon.

## Positioning

Unlike a static spreadsheet or a generic champion tier-list site, this
tool is opponent-specific (modeled on the actual opponent's historical
pick/ban tendencies), Fearless-series-aware (recommendations visibly shift
as the champion pool shrinks across Games 1-5), and shows the live
win-rate impact of each candidate move as the draft unfolds, rather than
a static pre-match report.

## Operating Context

Coach-facing prep tool, used before a match (not live during an actual
broadcast draft). Single surface: draft board + live win-rate meter +
insights panel, walked through linearly as a series' games are drafted.
Designed purely as a coach tool — hackathon judges encounter it through a
demo video/walkthrough, not as a second design audience the surface
itself needs to address.

## Capabilities and Constraints

- Cloud9 is always fixed as "our side"; the opponent is selectable.
- Series format (Bo1/Bo3/Bo5) is coach-configurable; Fearless Draft
  carryover (any champion picked by either team in a prior game becomes
  unavailable series-wide) is enforced automatically from Game 2 onward.
- Recommendation engine: a GA whose live fitness function is a neural net
  trained offline on historical draft-state -> win-probability data. A
  separate, explainable statistical blend (champion win rate, side delta,
  synergy, team history) is computed alongside it purely as the
  human-readable "why" — explicitly an approximation of the NN's
  reasoning, not literally it.
- Out of scope: ward/vision placement and jungle-pathing macro context —
  the GRID LoL event feed this project uses has no ward/vision events,
  only objective-timing events, which is insufficient to support that
  framing honestly.
- Data source: GRID esports data (`stl-hackathon-2026-grid-starter`),
  proprietary/licensed for the hackathon only — never committed to this
  repo (see `AGENTS.md`). External data (Riot Developer API for
  Diamond-Challenger solo queue stats, and/or Leaguepedia) is an optional
  stretch goal, surfaced only as a separate "current meta" badge; it never
  blocks or changes the core demo.
- Solo project; `main` is protected, changes land via PR (see
  `CONVENTIONS.md`).

## Evidence on Hand

GRID esports data repository (teams, players, characters, games, rosters,
and per-series event feeds for League of Legends), cloned locally and
referenced by path, never committed. No testimonials, case studies, or
press exist or should be fabricated. Full design rationale lives in
`docs/superpowers/specs/2026-11-19-lol-draft-coaching-assistant-design.md`.

## Product Principles

1. Plain-language clarity first, statistical accuracy second: every
   user-facing explanation favors a framing a non-technical coach can act
   on over one that's more precise but requires jargon.
2. Every recommendation shows its "why" — no bare score or move without a
   human-readable reason attached.
3. Fearless Draft carryover is a first-class mechanic, not an
   afterthought: the shrinking pool across a series is what makes
   Game 3/5 recommendations meaningfully different from Game 1.
4. Don't claim support for signals the data can't back — macro/ward
   context was cut rather than built on data that doesn't exist.
5. Core demo never depends on stretch goals (external meta data) being
   finished.

## Accessibility & Inclusion

Standard web accessibility baseline (sufficient contrast, keyboard
operability) — no further project-specific requirement established.
