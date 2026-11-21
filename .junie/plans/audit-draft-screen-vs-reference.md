# Impeccable Audit: Draft Screen vs. Reference Image

**Target:** `docs/design/champ_picker.png` (League of Legends Champion Select)
**Date:** 2026-11-21
**Components audited:** `DraftWorkspace.tsx`, `TeamColumn.tsx`, `ChampionSelector.tsx`, `ChampionTooltip.tsx`, `globals.css`

---

## Audit Health Score

| # | Dimension | Score | Key Finding |
|---|-----------|-------|-------------|
| 1 | Accessibility | 3 | Good a11y foundation; some missing ARIA labels and focus management |
| 2 | Performance | 3 | Memoized tiles and lazy images; some CSS could be optimized |
| 3 | Responsive Design | 2 | Desktop-only focus; mobile layout is minimal |
| 4 | Theming | 3 | Good token usage; some hard-coded values remain |
| 5 | Implementation Integrity | 2 | **Major gaps vs reference image** — team columns lack player avatars, roles, and champion portraits |
| **Total** | | **13/20** | **Acceptable (significant work needed)** |

**Rating band:** 10-13 Acceptable — significant work needed to match the reference image

---

## Implementation Integrity Verdict

**Status: FAIL** — The implementation does not express the reference image's visual design.

The current code implements a functional draft workspace with the correct three-column structure, but deviates significantly from `champ_picker.png` in visual fidelity. The reference shows a polished League of Legends champion select screen with player avatars, role icons, ban timers, and ornamental gold framing. The current implementation uses plain text lists for picks/bans without player identities, lacks the header timer, and misses the ornamental details that define the League aesthetic.

---

## Executive Summary

- **Audit Health Score:** **13/20** (Acceptable)
- **Total issues found:** P0: 3, P1: 5, P2: 4, P3: 3
- **Top critical issues:**
  1. Team columns lack player avatars, role icons, and champion portraits (P0)
  2. Header missing countdown timer shown in reference (P0)
  3. No ornamental gold framing matching League's aesthetic (P0)
  4. Champion grid layout doesn't match reference's compact portrait arrangement (P1)
  5. Footer missing game mode info and settings controls (P1)
- **Recommended next steps:** Rebuild TeamColumn with player avatars and role icons, add header timer, apply ornamental gold framing, and refine champion grid layout.

---

## Detailed Findings by Severity

### [P0] Team columns lack player avatars and role icons
- **Location:** `TeamColumn.tsx` lines 1-57
- **Category:** Implementation Integrity
- **Impact:** The reference image shows 5 player slots per team with circular avatars, role icons (TOP, SUPPORT, BOTTOM, MID, JUNGLE), and ban/pick status with timers. The current implementation shows only text labels for picks and bans without any player identity.
- **Evidence:** Reference image shows `Freckledman` (TOP), `BigBarrelConcern` (SUPPORT), `Benifo` (BOTTOM), `SonicScrewdriver` (MID), `Godfather75` (JUNGLE) with avatars and role icons. Current code renders `<li className="team-slot"><span>Next ban</span><small>#1</small></li>`.
- **Recommendation:** Rebuild TeamColumn to show 5 player slots with avatars, role icons, and ban/pick status. Use roster data from the backend endpoint.
- **Suggested command:** `/impeccable craft` — Rebuild team columns to match reference

### [P0] Header missing countdown timer
- **Location:** `DraftWorkspace.tsx` lines 98-106, `globals.css` lines 642-663
- **Category:** Implementation Integrity
- **Impact:** The reference image prominently displays "BAN A CHAMPION!" with a large "25" countdown timer below it. The current implementation shows only the action label without any timer.
- **Evidence:** Reference shows `BAN A CHAMPION!` with `25` in large numerals. Current code renders `<h1 className="action-label">BAN A CHAMPION!</h1>` without a timer.
- **Recommendation:** Add a countdown timer to the header, styled to match the reference's large numerals with gold accent.
- **Suggested command:** `/impeccable craft` — Add header timer matching reference

### [P0] No ornamental gold framing
- **Location:** `globals.css` lines 626-754
- **Category:** Implementation Integrity
- **Impact:** The reference image features ornamental gold framing around the header, champion grid, and team columns. The current implementation uses simple borders without the decorative elements.
- **Evidence:** Reference shows gold ornamental lines framing the header, circular gold border around the champion grid, and gold accents on active player slots. Current CSS uses `border-bottom: 2px solid var(--gold)` without ornamental details.
- **Recommendation:** Add ornamental gold framing using CSS pseudo-elements or SVG decorations to match the reference's League aesthetic.
- **Suggested command:** `/impeccable overdrive` — Add ornamental gold framing

### [P1] Champion grid layout doesn't match reference
- **Location:** `ChampionSelector.tsx` lines 244-271, `globals.css` lines 259-265
- **Category:** Implementation Integrity
- **Impact:** The reference shows a compact grid of champion portraits with names below, arranged in a circular/oval pattern. The current implementation uses a standard rectangular grid.
- **Evidence:** Reference shows 6-column grid with square portraits and names below. Current CSS uses `grid-template-columns: repeat(auto-fill, minmax(92px, 1fr))` which creates variable columns.
- **Recommendation:** Fix the champion grid to 6 columns with consistent portrait sizing matching the reference.
- **Suggested command:** `/impeccable layout` — Fix champion grid layout

### [P1] Footer missing game mode info and settings
- **Location:** `DraftWorkspace.tsx` lines 153-175, `globals.css` lines 736-753
- **Category:** Implementation Integrity
- **Impact:** The reference shows a bottom bar with game mode info ("5v5 RANKED FLEX"), settings icons, and chat controls. The current implementation only shows undo and finish buttons.
- **Evidence:** Reference shows `5v5 RANKED FLEX` with chat and settings icons. Current footer only has undo and finish buttons.
- **Recommendation:** Add game mode info and settings controls to the footer.
- **Suggested command:** `/impeccable craft` — Add footer game mode info

### [P1] Missing sort and role filter controls
- **Location:** `ChampionSelector.tsx` lines 233-243
- **Category:** Implementation Integrity
- **Impact:** The reference shows "Sort By Name" dropdown and role filter icons at the top of the champion grid. The current implementation only has a search input.
- **Evidence:** Reference shows sort dropdown and role filter icons. Current code renders only `<input type="search" placeholder="Search champions">`.
- **Recommendation:** Add sort dropdown and role filter icons to the champion selector toolbar.
- **Suggested command:** `/impeccable craft` — Add sort and role filter controls

### [P2] Champion portraits lack border styling
- **Location:** `globals.css` lines 312-342
- **Category:** Theming
- **Impact:** The reference shows champion portraits with distinct borders that change based on state (recommended, discouraged, etc.). The current implementation uses a simple border.
- **Evidence:** Reference shows gold borders for recommended champions, red borders for discouraged. Current CSS uses `border: 1px solid #6b343b` for all portraits.
- **Recommendation:** Add state-based border styling to champion portraits.
- **Suggested command:** `/impeccable colorize` — Add state-based portrait borders

### [P2] Active player slot highlighting doesn't match reference
- **Location:** `globals.css` lines 726-730
- **Category:** Theming
- **Impact:** The reference shows the active player slot with a gold highlight and timer. The current implementation uses a simple border color change.
- **Evidence:** Reference shows gold background and timer on active slot. Current CSS uses `border-color: var(--gold)` and `box-shadow`.
- **Recommendation:** Update active slot styling to match reference's gold highlight and timer display.
- **Suggested command:** `/impeccable polish` — Refine active slot styling

### [P2] Missing chat log in team columns
- **Location:** `TeamColumn.tsx`
- **Category:** Implementation Integrity
- **Impact:** The reference shows a chat log at the bottom of the left team column with player join messages. The current implementation has no chat functionality.
- **Evidence:** Reference shows `SonicScrewdriver joined the lobby` messages. Current code has no chat component.
- **Recommendation:** Add a chat log section to the team columns.
- **Suggested command:** `/impeccable craft` — Add chat log to team columns

### [P2] Team column background doesn't match reference
- **Location:** `globals.css` lines 689-697
- **Category:** Theming
- **Impact:** The reference shows team columns with a dark background and subtle gradient. The current implementation uses a flat surface color.
- **Evidence:** Reference shows dark background with subtle gradients. Current CSS uses `background: var(--surface)`.
- **Recommendation:** Add subtle gradients to team columns to match reference.
- **Suggested command:** `/impeccable colorize` — Add team column gradients

### [P3] Focus indicator could be more distinctive
- **Location:** `globals.css` lines 43-46
- **Category:** Accessibility
- **Impact:** The current focus indicator uses a yellow outline. The reference uses gold accents for active states.
- **Recommendation:** Align focus indicator with the gold accent color scheme.
- **Suggested command:** `/impeccable polish` — Refine focus indicators

### [P3] Champion tooltip styling could match reference
- **Location:** `globals.css` lines 362-385
- **Category:** Theming
- **Impact:** The tooltip uses a simple dark background with gold border. The reference shows more elaborate tooltips with champion details.
- **Recommendation:** Enhance tooltip styling to match reference's elaborate design.
- **Suggested command:** `/impeccable polish` — Enhance tooltip styling

### [P3] Missing "FIRST PICK" label in header
- **Location:** `DraftWorkspace.tsx` lines 98-106
- **Category:** Implementation Integrity
- **Impact:** The reference shows "FIRST PICK" in the top right corner of the header. The current implementation doesn't show this label.
- **Recommendation:** Add the current pick/ban phase label to the header.
- **Suggested command:** `/impeccable polish` — Add phase label to header

---

## Patterns & Systemic Issues

1. **Visual fidelity gap:** The implementation prioritizes functionality over visual match with the reference. The three-column structure is correct, but the ornamental details, player avatars, and decorative elements are missing.

2. **Hard-coded values:** Some CSS values (like portrait border colors) are hard-coded rather than using design tokens.

3. **Missing League aesthetic:** The reference's ornamental gold framing, circular borders, and decorative elements are absent, making the implementation feel more like a generic dashboard than a League champion select screen.

---

## Positive Findings

1. **Correct three-column layout:** The implementation correctly uses the three-column structure with team columns flanking the champion picker.

2. **Good accessibility foundation:** ARIA labels, keyboard navigation, and focus management are properly implemented.

3. **Proper token usage:** CSS custom properties and Tailwind tokens are used consistently for colors.

4. **Champion portraits with fallbacks:** The implementation correctly handles portrait loading with graceful fallbacks.

5. **No-scroll layout:** The implementation correctly prevents page scrolling at desktop sizes.

---

## Recommended Actions

1. **[P0] `/impeccable craft`**: Rebuild TeamColumn with player avatars, role icons, and champion portraits to match reference
2. **[P0] `/impeccable craft`**: Add countdown timer to header matching reference's large numerals
3. **[P0] `/impeccable overdrive`**: Add ornamental gold framing around header, champion grid, and team columns
4. **[P1] `/impeccable layout`**: Fix champion grid to 6-column layout with consistent portrait sizing
5. **[P1] `/impeccable craft`**: Add sort dropdown and role filter controls to champion selector
6. **[P1] `/impeccable craft`**: Add game mode info and settings controls to footer
7. **[P2] `/impeccable colorize`**: Add state-based border styling to champion portraits
8. **[P2] `/impeccable polish`**: Refine active slot styling to match reference
9. **[P3] `/impeccable polish`**: Add phase label to header and enhance tooltip styling

You can ask me to run these one at a time, all at once, or in any order you prefer.

Re-run `/impeccable audit` after fixes to see your score improve.
