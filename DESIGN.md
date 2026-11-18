<!-- SEED: established with the user before implementation; re-run /impeccable document once there's code to capture the actual tokens and components. -->

---
name: Draft Coach
description: A broadcast-grade pick/ban readout for Cloud9's League of Legends coaching staff.
---

# Design System: Draft Coach

## Overview

**Creative North Star: "The Broadcast Sequence Bar"**

Draft Coach reads as an LCS/LEC on-air draft graphic repurposed as a working
coaching instrument, not a SaaS analytics dashboard wearing esports colors.
The draft board is the broadcast sequence bar itself: pick/ban order runs
as a horizontal sequence across the top, the predicted win-rate reads like
an on-air score ticker, and the recommendation's plain-language reason
renders as a broadcast lower-third callout. Everything the coach needs is
visible on one continuous screen — nothing lives behind a tab or modal.

The material character is flat broadcast-graphic chrome: near-black ground,
hairline rules, crisp rectangular chips, no soft shadows or glassmorphism.
Motion is a single signature interaction — a lock-in visibly moves the
ticker and updates the lower-third, crisp and immediate like a live
broadcast graphic updating, never a lingering animation.

Confirmed visual rejections: no generic SaaS card-grid dashboard (soft
shadows, blue gradients); no generic dark-theme neon-esports cliché
(Orbitron/Rajdhani-style futurism); no literal reproduction of the real
League Client champion-select screen.

**Key Characteristics:**
- Broadcast sequence bar as primary navigation, not a sidebar
- Near-black ground with one saturated accent per side, not a gradient
- Flat chrome: hairlines and contrast carry depth, not shadows
- One dominant, uncluttered win-rate numeral — the ticker owns the field
- Every recommendation ships with a plain-language reason, rendered as a
  broadcast lower-third, never a bare score

## Colors

Color strategy: **Committed** — one saturated accent per side carries real
surface share against a near-black broadcast ground; this is not a
restrained neutrals-plus-accent system, because the broadcast world this
direction borrows from is inherently high-contrast and accent-led.

### Primary
- **Cloud9 accent** [to be resolved during implementation]: Cloud9's own
  side of the sequence bar, the champion grid's active-team framing, and
  the primary action states. Cloud9 is always present (fixed as "our
  side"), so this is a fixed brand accent, not a rotating team color.

### Secondary
- **Opponent accent** [to be resolved during implementation]: the
  opponent's side of the sequence bar and grid framing. Deliberately
  distinct from the Cloud9 accent and from the two semantic state colors
  below, since the opponent varies by match and should never be confused
  with a status color.

### Neutral
- **Broadcast ground** [to be resolved during implementation]: near-black
  base for the whole surface.
- **Hairline / divider** [to be resolved during implementation]: the rule
  lines that carry structure in place of shadows or cards.

### Semantic (state, not brand)
- **Fearless-excluded** [to be resolved during implementation]: dim-red,
  broadcast "cancelled row" treatment for champions excluded by series
  carryover.
- **Limited-data flag** [to be resolved during implementation]: amber,
  broadcast "delay" treatment for an opponent read built on too few games
  — flagged explicitly per the product's clarity-first principle, never
  silently blended away.

### Named Rules
**The One Numeral Rule.** The live win-rate ticker is the one place
numbers are allowed to dominate the viewport. Every other numeric readout
(contributing-factor scores, tendency percentages) stays visually
subordinate to it.

## Typography

**Display/Label Font:** condensed broadcast sans [to be resolved during
implementation] — carries the sequence bar, ticker labels, and chips.

**Mono Font:** [to be resolved during implementation] — drives the
win-rate ticker's numerals and any other broadcast-style figure readout.

**Body Font:** a highly legible humanist sans [to be resolved during
implementation], distinct from the display face — the lower-third's
plain-language explanations are the product's trust mechanism and must
never inherit the display face's condensed, headline-first character.

**Character:** broadcast-headline urgency up top, plain-spoken clarity in
the explanatory copy underneath it. The pairing should feel like a live
broadcast graphic that still respects a reader who has no technical
background.

### Named Rules
**The Exact-Phrasing Rule.** Contributing-factor explanations read as
hand-checked and specific ("They've answered this ban with Sejuani 70% of
the time"), never as generic templated copy or raw statistical notation.

## Layout

Single continuous screen, no setup → draft → summary route change. Three
persistent zones, always visible together: the sequence bar (top, full
width, wayfinding for where the draft stands for both teams), the champion
grid plus win-rate ticker (center), and the insights lower-third (bottom
or adjacent, never hidden behind a tab or modal). Exact grid, container,
and responsive breakpoint behavior: [to be resolved during implementation].

## Elevation & Depth

Flat by design — no shadows, no glassmorphism, no tonal elevation layers.
Depth and hierarchy come entirely from the near-black ground, one
saturated accent per side, and hairline rules, matching the broadcast
graphic's own flat-chrome convention.

### Named Rules
**The Flat-Chrome Rule.** No `box-shadow`, no blur, no elevated surfaces
at rest or on hover. Structure is drawn with hairlines and contrast, never
implied with a shadow.

## Shapes

Crisp rectangular chips and blocks — the broadcast sequence bar, pick/ban
slots, and chips carry minimal or no corner rounding, in line with the
broadcast-graphic signature rather than the rounded-card language of a
generic SaaS dashboard. A single clipped or angled corner is reserved for
the sequence bar's "active now" indicator only — never used decoratively
elsewhere.

## Do's and Don'ts

### Do:
- **Do** let the win-rate ticker be the one dominant, uncluttered numeral
  on screen (The One Numeral Rule).
- **Do** give the pick/ban clock an exact, measured timer treatment, not
  a decorative countdown animation.
- **Do** write every contributing-factor explanation as specific and
  hand-checked (The Exact-Phrasing Rule), never generic template copy.
- **Do** commit to one disciplined type system with hairline dividers
  throughout, rather than mixing weights or styles loosely.

### Don't:
- **Don't** use a generic SaaS card-grid dashboard look (soft shadows,
  blue gradients, rounded cards).
- **Don't** use the generic dark-theme neon-esports cliché (Orbitron/
  Rajdhani-style futuristic gaming fonts).
- **Don't** reproduce the real League Client champion-select screen
  literally.
- **Don't** add shadows, blur, or elevated surfaces anywhere (The
  Flat-Chrome Rule).
