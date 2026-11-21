---
sessionId: session-261120-171115-196e
---

# Requirements

### Goal and layout
Completely redesign the draft screen to match `docs/design/champ_picker.png` exactly, focusing only on the core draft elements: ban/pick actions, champion picker, and minimal analysis. Remove all unnecessary elements like coaching strips, stats panels, series history, and detailed insights.

- Match the reference image's clean League-style champion select: header with "BAN A CHAMPION!" and timer, left team column with player roles/bans, central champion grid with portraits, right team column with bans, and minimal bottom bar.
- Show all ten bans and ten picks in the two team columns with the active draft action identifiable. Retain the actual pro `DRAFT_ORDER`, Fearless exclusions, undo, and game-result recording.
- Keep champion search and the scrollable portrait grid. Remove role filter buttons, coaching insights, win-rate charts, and series history from the main view.
- Use champion portraits where available, with graceful fallbacks. No fabricated timers, rosters, or unnecessary elements.

### Constraints
- Preserve keyboard operation, visible focus, sufficient contrast, and accessible hover/focus explanations.
- No changes to the recommendation model, draft rules or underlying analysis. Do not copy or commit licensed GRID data.

# Technical Design

### Existing state
- `frontend/components/DraftWorkspace/DraftWorkspace.tsx` renders a complex layout with TopBar, TeamColumns, ChampionSelector, CoachingStrip, stats panel, and series history. The reference image is much simpler.
- `ChampionSelector.tsx` has role filters, search, and champion grid with tooltips. The reference shows only search/sort at the top.
- `TeamColumn.tsx` shows picks, bans, and historical roster. The reference shows player avatars with role icons and ban/pick slots.

### Design and data decisions
- Simplify `DraftWorkspace.tsx` to match the reference: header with action label and timer, left team column, central champion grid, right team column, minimal bottom bar. Remove CoachingStrip, stats panel, series history, and TopBar.
- Simplify `ChampionSelector.tsx` to remove role filter buttons, keeping only search and the scrollable champion grid with portraits.
- Simplify `TeamColumn.tsx` to show player avatars with role icons and ban/pick slots with champion portraits, matching the reference's clean style.
- Update `globals.css` to match the reference's dark theme with red accents, gold framing, and clean typography.

### File structure
- Modify `frontend/components/DraftWorkspace/DraftWorkspace.tsx` - simplify layout to match reference
- Modify `frontend/components/DraftWorkspace/ChampionSelector.tsx` - remove role filters, keep search and grid
- Modify `frontend/components/DraftWorkspace/TeamColumn.tsx` - simplify to show player avatars and ban/pick slots
- Modify `frontend/app/globals.css` - update styling to match reference
- Remove `frontend/components/DraftWorkspace/CoachingStrip.tsx` - not needed in reference
- Remove `frontend/components/DraftWorkspace/TopBar.tsx` - not needed in reference

# Testing

### Validation
- Run frontend Vitest tests and TypeScript checks to ensure no breaking changes.
- Verify the simplified layout matches the reference image at 1280×720 and 1920×1080.
- Test champion grid scrolling, search functionality, and tooltip behavior.
- Verify team columns show bans/picks correctly with active slot highlighting.
- Ensure keyboard navigation and focus visibility work correctly.

# Delivery Steps

### ✓ Step 1: Simplify the draft workspace layout
Remove all unnecessary elements to match the reference image exactly.

- Remove `CoachingStrip.tsx` and `TopBar.tsx` from `DraftWorkspace.tsx`.
- Remove the stats panel, insights section, and series history from the layout.
- Simplify the layout to: header with action label, left team column, central champion grid, right team column, minimal bottom bar.
- Update `globals.css` to match the reference's dark theme with red accents and gold framing.

### ✓ Step 2: Simplify the champion selector
Remove role filters and unnecessary elements to match the reference.

- Remove role filter buttons from `ChampionSelector.tsx`.
- Keep only search input and the scrollable champion grid with portraits.
- Simplify the toolbar to match the reference's clean style.

### ✓ Step 3: Simplify the team columns
Match the reference's clean player avatar and ban/pick slot style.

- Simplify `TeamColumn.tsx` to show player avatars with role icons and ban/pick slots.
- Remove historical roster section and display-only swap functionality.
- Style ban/pick slots to match the reference's clean appearance.

### ✓ Step 4: Restructure layout to match redesign_overall.png
Rebuild the entire workspace to match `docs/design/redesign_overall.png` exactly.

- Champion selector on the left with role icons, search, and scrollable grid.
- Stats + guidance panel on the right with win rate graph and reasoning.
- Draft bar at the bottom showing both teams' bans/picks with player names and team logos.
- Remove the League-style two-team-columns layout entirely.
- Update DESIGN.md to reflect the new layout.

### ✓ Step 5: Remove all halo effects and gradients
Strip all box-shadow glows and gradients except for hover effects and banners.

- Remove halo effects from recommended champions, tooltips, active slots, player avatars.
- Remove gradient backgrounds from header, team columns, tooltips.
- Keep only flat colors and borders for all elements.
- Allow gradients only for hover states and banner elements.

### ✓ Step 6: Remove the countdown timer
Remove the unnecessary countdown timer from the header.

- Remove timer state, effects, and display from `DraftWorkspace.tsx`.
- Simplify header to show only action label and active team.

### ✓ Step 7: Fix tooltip visibility for all recommended/discouraged champions
Ensure tooltips appear consistently for good/bad champion selections.

- Verify tooltip portal positioning works for all champions in the grid.
- Ensure tooltips show for recommended and discouraged champions on hover.

### ✓ Step 8: Verify and test
All tests pass, TypeScript compiles clean.

- 52 frontend tests pass (Vitest), TypeScript compiles clean.
- Layout matches `redesign_overall.png` structure.

### ✓ Step 9: Add resizable panel sections and change title
Add drag-to-resize handles between the three sections (champion selector, stats panel, draft bar) and change the header title to "Draft Recommendation".

- Add resize handle components between the champion selector and stats panel (vertical divider) and between the main area and draft bar (horizontal divider).
- Implement mouse drag handlers in `DraftWorkspace.tsx` to adjust panel widths/heights.
- Update `globals.css` with resize handle styling and dynamic panel sizing using CSS custom properties.
- Change header title from action label to "Draft Recommendation".

### ✓ Step 10: Fix win rate graph size and appearance
Make the win rate graph larger and more readable in the stats panel.

- Increase SVG dimensions from 120×32 to 400×150 in `StatsPanel.tsx`.
- Add axis labels, grid lines, and data point tooltips to the chart.
- Update `globals.css` `.win-rate-chart` class to match the new size.
- Add proper Y-axis labels (0-100%) and X-axis labels (pick numbers).

### ✓ Step 11: Add adaptive ban/pick chain recommendations
Extend the backend to suggest full ban-pick chains, not just the next move.

- Add `chain_recommendations` field to `DraftResponse` model in `models.py` containing a sequence of recommended ban/pick moves.
- Implement `recommend_chain()` method in `DraftEngine` that simulates the remaining draft slots, scoring each move.
- Use the existing GA (genetic algorithm) or statistical scoring to evaluate chain outcomes.
- Update the frontend `StatsPanel` to display the chain of recommended moves with visual indicators.

### ✓ Step 12: Enable GPU acceleration for the model and continuous calculation
Make the model use GPUs when available and keep recommendations evolving.

- Update `WinPredictor` to use CUDA tensors when GPU is available in `backend/app/model/predictor.py`.
- Add a background calculation loop in the backend that continuously re-evaluates recommendations as the draft state changes.
- Use WebSocket or server-sent events to push updated recommendations to the frontend.
- The calculation stops when the user makes a move (which triggers a fresh request) or explicitly cancels.

### ✓ Step 13: Resolve DESIGN.md placeholders from actual code
Fill in all `[to be resolved during implementation]` placeholders with real values from `tailwind.config.ts` and `globals.css`.

- Replace color placeholders with actual Tailwind tokens (ground `#080708`, Cloud9 `#74b9e5`, opponent `#d85c62`, etc.).
- Replace font placeholders with actual font families (display: Arial Narrow, body: Arial, mono: IBM Plex Mono).
- Update the layout description to match the current implementation exactly.

### ✓ Step 14: Update stale documentation
Update CLAUDE.md, README.md, and other files with outdated status information.

- Update CLAUDE.md Status section: frontend and model are built, not planned.
- Update README.md repository map: remove `(planned)` from `frontend/`.
- Update README.md roadmap: mark all 4 items as complete.
- Delete empty `docs/design/UI_redesign.md` file.

### ✓ Step 15: Strengthen conventions and rules
Clarify and strengthen statements in CONVENTIONS.md, AGENTS.md, and DESIGN.md.

- Strengthen the halo & gradient rule in DESIGN.md.
- Clarify the auto-close issue convention in CONVENTIONS.md and AGENTS.md.
- Ensure all design rules are consistent across files.
- Add No-Scroll Rule and Three-Zone Rule to DESIGN.md.
- Add design token usage rules to Do's/Don'ts.

### ✓ Step 16: Verify and commit
Run tests and commit all documentation changes.

- Run frontend and backend tests to ensure no breaking changes.
- Commit all documentation changes with proper Conventional Commit message.
- Push to a new branch and create a PR.