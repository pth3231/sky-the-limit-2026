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
- **Cloud9 accent** `#74b9e5` (`draft.cloud9`): Cloud9's own side of the
  sequence bar, the champion grid's active-team framing, and the primary
  action states. Cloud9 is always present (fixed as "our side"), so this
  is a fixed brand accent, not a rotating team color.

### Secondary
- **Opponent accent** `#d85c62` (`draft.opponent`): the opponent's side of
  the sequence bar and grid framing. Deliberately distinct from the Cloud9
  accent and from the two semantic state colors below, since the opponent
  varies by match and should never be confused with a status color.

### Neutral
- **Broadcast ground** `#080708` (`draft.ground`): near-black base for the
  whole surface.
- **Surface** `#151114` (`draft.surface`): elevated panels and cards.
- **Raised** `#21191b` (`draft.raised`): hover states and active elements.
- **Hairline / divider** `#493b34` (`draft.line`): the rule lines that
  carry structure in place of shadows or cards.
- **Text** `#f1ece5` (`draft.text`): primary text color.
- **Muted** `#aaa099` (`draft.muted`): secondary text and disabled states.
- **Gold** `#c5a36a` (`draft.gold`): active states, accents, and highlights.

### Semantic (state, not brand)
- **Fearless-excluded** `#934951` (`draft.excluded`): dim-red, broadcast
  "cancelled row" treatment for champions excluded by series carryover.
- **Limited-data flag** `#e2bf78` (`draft.warning`): amber, broadcast
  "delay" treatment for an opponent read built on too few games — flagged
  explicitly per the product's clarity-first principle, never silently
  blended away.

### Named Rules
**The One Numeral Rule.** The live win-rate ticker is the one place
numbers are allowed to dominate the viewport. Every other numeric readout
(contributing-factor scores, tendency percentages) stays visually
subordinate to it.

## Typography

**Display/Label Font:** `Arial Narrow`, `Liberation Sans Narrow`, `Arial`,
`sans-serif` (`fontFamily.display`) — a condensed broadcast sans that
 carries the sequence bar, ticker labels, and chips.

**Mono Font:** `IBM Plex Mono`, `Consolas`, `monospace`
(`fontFamily.mono`) — drives the win-rate ticker's numerals and any other
broadcast-style figure readout.

**Body Font:** `Arial`, `Helvetica`, `sans-serif` (`fontFamily.body`) — a
highly legible humanist sans, distinct from the display face. The
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
1920×1080). Three persistent zones: a minimal header with the title
"Draft Recommendation", a two-column main area with the champion selector
on the left and the stats + guidance panel on the right, and a compact
draft bar at the bottom showing both teams' bans and picks. The champion
grid and stats panel scroll internally. Panels are resizable via drag
handles at their boundaries. No hidden panels, no modals, no drawers —
everything is visible at once.

### Named Rules
**The No-Scroll Rule.** The page body must not scroll at supported desktop
sizes (1280×720 minimum). All scrolling happens within individual panels
(champion grid, stats panel). The layout uses `100dvh` with `overflow: hidden`
on the page root.

**The Three-Zone Rule.** The screen is divided into exactly three zones:
champion selector, stats + guidance, and draft bar. No additional panels,
drawers, or hidden sections may be added without explicit product approval.

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
- **Do** keep the header minimal: "Draft Recommendation" title and action label.
- **Do** use halo effects and gradients for hover states and banner elements
  to provide visual feedback (The Halo & Gradient Rule).
- **Do** maintain the three-zone layout without page scroll (The No-Scroll Rule).
- **Do** use design tokens from `tailwind.config.ts` — never hard-code colors
  in components.

### Don't:
- **Don't** add unnecessary panels, drawers, or hidden sections (The
  Three-Zone Rule).
- **Don't** use a generic SaaS card-grid dashboard look (soft shadows,
  blue gradients, rounded cards).
- **Don't** use the generic dark-theme neon-esports cliché (Orbitron/
  Rajdhani-style futuristic gaming fonts).
- **Don't** add halo effects, gradients, or glow to static UI elements
  (The Halo & Gradient Rule).
- **Don't** add shadows, blur, or elevated surfaces anywhere (The
  Flat-Chrome Rule).
- **Don't** allow the page body to scroll at supported desktop sizes
  (The No-Scroll Rule).
- **Don't** invent roles, player names, or roster facts from row order
  or assumptions.
- **Don't** commit GRID data or derived datasets to version control.
- **Don't** hard-code color values in components — use Tailwind design
  tokens (`draft.ground`, `draft.gold`, etc.).
