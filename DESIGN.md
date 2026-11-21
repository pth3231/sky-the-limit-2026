---
name: Draft Coach
description: A focused draft workspace with champion selection, analysis, and draft status.
---

# Design System: Draft Coach

## Overview

**Creative North Star: "Focused Draft Workspace"**

Draft Coach is a single-screen draft workspace matching `docs/design/redesign_overall.png`.
The layout has two main zones: a champion selector on the left with role icons,
search, and a scrollable portrait grid, and a stats + guidance panel on the right
showing win rate trends and AI-driven reasoning. A compact draft bar at the bottom
displays both teams' bans and picks with player names.

The material character is flat and functional: near-black surfaces, hairline rules,
blue/red team cues, and restrained gold accents for active states. No halo effects,
no gradients, no soft shadows — only flat colors and borders. Gradients and glow
effects are reserved exclusively for hover states and banner elements.

**Key Characteristics:**
- Two-column layout: champion selector left, stats + guidance right
- Compact draft bar at the bottom showing both teams' bans/picks
- Minimal header with current action label
- Flat surfaces with hairline rules — no halo effects or gradients at rest
- Gold accents only for active states and hover effects
- Gradients and glow effects permitted only for hover states and banners

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

Single continuous screen, no scrolling at desktop sizes (1280×720 and
1920×1080). Three persistent zones: a minimal header with the current
action label, a two-column main area with the champion selector on the
left and the stats + guidance panel on the right, and a compact draft bar
at the bottom showing both teams' bans and picks. The champion grid and
stats panel scroll internally. No hidden panels, no modals, no drawers —
everything is visible at once.

## Elevation & Depth

Flat by design — no shadows, no glassmorphism, no tonal elevation layers,
no halo effects, no gradients at rest. Depth and hierarchy come entirely
from the near-black ground, saturated accents, and hairline rules.

### Named Rules
**The Flat-Chrome Rule.** No `box-shadow`, no blur, no elevated surfaces,
no halo effects, no gradients — except for hover states and banner
elements. Structure is drawn with hairlines and contrast, never implied
with a shadow or glow.

**The Halo & Gradient Rule.** Halo effects (`box-shadow` with color spread)
and gradients are permitted only for hover effects and banner elements.
All other UI elements use flat colors and borders exclusively.

## Shapes

Crisp rectangular chips and blocks — the broadcast sequence bar, pick/ban
slots, and chips carry minimal or no corner rounding, in line with the
broadcast-graphic signature rather than the rounded-card language of a
generic SaaS dashboard. A single clipped or angled corner is reserved for
the sequence bar's "active now" indicator only — never used decoratively
elsewhere.

## Do's and Don'ts

### Do:
- **Do** keep the layout focused: ban/pick, analysis, and champion selection only.
- **Do** use permitted Riot Data Dragon portraits with graceful fallbacks.
- **Do** write every contributing-factor explanation as specific and
  hand-checked (The Exact-Phrasing Rule), never generic template copy.
- **Do** keep the header and footer minimal and unobtrusive.
- **Do** use halo effects and gradients for hover states and banner elements
  to provide visual feedback (The Halo & Gradient Rule).

### Don't:
- **Don't** add unnecessary panels, drawers, or hidden sections.
- **Don't** use a generic SaaS card-grid dashboard look (soft shadows,
  blue gradients, rounded cards).
- **Don't** use the generic dark-theme neon-esports cliché (Orbitron/
  Rajdhani-style futuristic gaming fonts).
- **Don't** add halo effects, gradients, or glow to static UI elements
  (The Halo & Gradient Rule).
- **Don't** add shadows, blur, or elevated surfaces anywhere (The
  Flat-Chrome Rule).
- **Don't** invent roles, player names, or roster facts from row order
  or assumptions.
- **Don't** commit GRID data or derived datasets to version control.
