# Oracle's Elixir Source Parser (Milestone 2) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add Oracle's Elixir (OE) pro-match CSVs as a second training-data source, normalized into the existing `GameDraft` / `DraftAction` / `PlayerLine` shape so `features.py` and `train.py` stay source-agnostic.

**Architecture:** A new `app/pipeline/sources/` package with `oracles_elixir.py`. OE's match CSV has one row per participant (10 player rows + 2 team-summary rows per game). The parser groups rows by `gameid`, builds each complete game into a `GameDraft` whose `actions` follow the canonical `DRAFT_ORDER` sequence (OE records no draft order, so it is synthesized positionally), resolves OE champion **names** to GRID `character_id`s so both sources share one embedding vocabulary, and coerces missing numeric fields to zero. A manual build CLI merges OE games into the gitignored `.derived/drafts.json`.

**Tech Stack:** Python 3.12, stdlib `csv`, Pydantic models (reuse `GameDraft`/`DraftAction`/`PlayerLine`), `app.engine.draft_order.DRAFT_ORDER`, `app.pipeline.ddragon.normalize_champion_name`.

## Global Constraints

- Normalize into the **unchanged** `GameDraft` / `DraftAction` / `PlayerLine` models from `app/pipeline/dataset.py` and `app/pipeline/drafts.py`. Do not alter those models.
- **Champion identity:** map OE champion display names to GRID `character_id` via `normalize_champion_name`. Build the map from GRID `Champion` metadata (`normalize_champion_name(champion.name) -> champion.character_id`). Unmatched names fall back to the canonical key `normalize_champion_name(name)` and are collected for logging — never fatal.
- **Draft-order alignment is mandatory:** the training/scoring code indexes `DRAFT_ORDER[i]` against `actions[i]` (`app/model/train.py` `_scorer_probabilities`, `_prefix_examples`). OE games must therefore emit the full 20-slot sequence in `DRAFT_ORDER` order. A game missing any of its 5 picks/side or 5 bans/side is **skipped and logged**, never partially emitted.
- **Missing numeric fields default to 0.0** via a local `_number()` helper mirroring `app/pipeline/drafts.py` (`float(value)` guarded by `except (TypeError, ValueError)`).
- **Never commit raw OE data.** Tests use hand-written synthetic CSV rows only. The build CLI reads a local path the user supplies and writes only into the gitignored `.derived/`.
- TDD per `CONVENTIONS.md`; `ruff check .`, `ruff format --check .`, `basedpyright`, `pytest -q` (run from `backend/`) stay green on every commit. Stage files explicitly; one PR, `Closes #43`.

## File Structure

- Create `backend/app/pipeline/sources/__init__.py` (empty package marker).
- Create `backend/app/pipeline/sources/oracles_elixir.py` (parser + build CLI).
- Create `backend/tests/pipeline/test_oracles_elixir.py` (synthetic-fixture tests).

### Assumed OE CSV columns (implementer verifies header against a real OE file)

One row per participant. Columns used (all read as strings via `csv.DictReader`):

`gameid`, `datacompleteness`, `date`, `patch`, `side` (`Blue`/`Red`), `position` (`top`/`jng`/`mid`/`bot`/`sup`/`team`), `teamname`, `teamid`, `champion` (player rows), `ban1`..`ban5` (team rows), `result` (`1`/`0`), `kills`, `assists`, `teamkills`, `damageshare`, `totaldamagetaken`, `visionscore`, `earnedgoldshare`.

The two `position == "team"` rows per game carry bans + `result`; the ten player rows carry `champion` + per-player stats.

---

### Task 1: Position/role map and champion-id resolver

**Files:**
- Create: `backend/app/pipeline/sources/__init__.py`
- Create: `backend/app/pipeline/sources/oracles_elixir.py`
- Test: `backend/tests/pipeline/test_oracles_elixir.py`

**Interfaces:**
- Consumes: `app.pipeline.ddragon.normalize_champion_name`, `app.pipeline.meta.Champion`, `app.pipeline.drafts.Role`.
- Produces:
  - `OE_POSITION_ROLES: dict[str, Role]`
  - `build_champion_id_map(champions: Mapping[str, Champion]) -> dict[str, str]`
  - `resolve_champion_id(name: str, name_to_id: Mapping[str, str]) -> str`
  - `_number(value: object) -> float`

- [ ] **Step 1: Write the failing test**

```python
from collections.abc import Mapping

from app.pipeline.meta import Champion
from app.pipeline.sources.oracles_elixir import (
    OE_POSITION_ROLES,
    _number,
    build_champion_id_map,
    resolve_champion_id,
)


def _champions() -> Mapping[str, Champion]:
    return {
        "Aatrox": Champion(character_id="grid-aatrox", name="Aatrox", games_played=1),
        "LeeSin": Champion(character_id="grid-leesin", name="Lee Sin", games_played=1),
    }


def test_position_roles_cover_oe_abbreviations() -> None:
    assert OE_POSITION_ROLES == {
        "top": "top",
        "jng": "jungle",
        "mid": "mid",
        "bot": "bot",
        "sup": "support",
    }


def test_resolve_champion_id_maps_names_to_grid_ids_with_fallback() -> None:
    name_to_id = build_champion_id_map(_champions())
    assert resolve_champion_id("Aatrox", name_to_id) == "grid-aatrox"
    # Normalization ignores spaces/case, so OE "Lee Sin" matches GRID "Lee Sin".
    assert resolve_champion_id("Lee Sin", name_to_id) == "grid-leesin"
    # Unknown champion falls back to its normalized key, never raising.
    assert resolve_champion_id("Smolder", name_to_id) == "smolder"


def test_number_coerces_blanks_to_zero() -> None:
    assert _number("12.5") == 12.5
    assert _number("") == 0.0
    assert _number(None) == 0.0
    assert _number("n/a") == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && uv run pytest tests/pipeline/test_oracles_elixir.py -q`
Expected: FAIL with `ModuleNotFoundError: app.pipeline.sources.oracles_elixir`.

- [ ] **Step 3: Create the package marker and implement the helpers**

Create `backend/app/pipeline/sources/__init__.py` as an empty file.

Create `backend/app/pipeline/sources/oracles_elixir.py`:

```python
"""Parse Oracle's Elixir match CSVs into the pipeline's GameDraft shape."""

from collections.abc import Mapping

from app.pipeline.drafts import Role
from app.pipeline.ddragon import normalize_champion_name
from app.pipeline.meta import Champion

OE_POSITION_ROLES: dict[str, Role] = {
    "top": "top",
    "jng": "jungle",
    "mid": "mid",
    "bot": "bot",
    "sup": "support",
}


def _number(value: object) -> float:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0.0


def build_champion_id_map(champions: Mapping[str, Champion]) -> dict[str, str]:
    """Map normalized champion display names to GRID character ids."""
    return {
        normalize_champion_name(champion.name): champion.character_id
        for champion in champions.values()
    }


def resolve_champion_id(name: str, name_to_id: Mapping[str, str]) -> str:
    """Resolve an OE champion name to a GRID id, falling back to its normalized key."""
    normalized = normalize_champion_name(name)
    return name_to_id.get(normalized, normalized)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && uv run pytest tests/pipeline/test_oracles_elixir.py -q`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add app/pipeline/sources/__init__.py app/pipeline/sources/oracles_elixir.py tests/pipeline/test_oracles_elixir.py
git commit -m "feat: add Oracle's Elixir champion-id resolver and role map"
```

---

### Task 2: Build one game's grouped rows into a GameDraft

**Files:**
- Modify: `backend/app/pipeline/sources/oracles_elixir.py`
- Test: `backend/tests/pipeline/test_oracles_elixir.py`

**Interfaces:**
- Consumes: Task 1 helpers, `app.engine.draft_order.DRAFT_ORDER`, `app.pipeline.dataset.GameDraft`, `app.pipeline.drafts.DraftAction`/`PlayerLine`.
- Produces: `build_game(rows: list[Mapping[str, str]], name_to_id: Mapping[str, str]) -> GameDraft | None` — returns `None` (with no raise) for an incomplete/unparseable game.

**Behavior:**
- Split `rows` into player rows (`position != "team"`, lowercased) and the two team rows (`position == "team"`).
- Require exactly one `Blue` and one `Red` team row, and for each side exactly 5 player rows whose `position` is in `OE_POSITION_ROLES` and 5 non-blank bans (`ban1`..`ban5`). Otherwise return `None`.
- Team ids: `teamid` if non-blank else `normalize_champion_name(teamname)`; keep blue/red/winner consistent within the game. Winner = the side whose team row has `result == "1"`.
- Picks per side: player rows sorted by canonical position order `("top", "jng", "mid", "bot", "sup")`.
- `actions`: iterate `DRAFT_ORDER`; keep a ban cursor and a pick cursor **per side**; each slot draws the next champion from that side's ban or pick list. `sequence_number = slot.index + 1`, `team_id` = that side's team id, `champion_id` = resolved id, `champion_name` = original OE name.
- `lineups`: one `PlayerLine` per player row (`team_id`, `role` from `OE_POSITION_ROLES`, resolved `champion_id`, `damage_share=_number(damageshare)`, `damage_taken=_number(totaldamagetaken)`, `vision_score=_number(visionscore)`, `gold_share=_number(earnedgoldshare)`, `kill_participation=(kills+assists)/teamkills` guarded against zero).
- `game_number = 1`, `series_id = gameid`, `finished = True`.

- [ ] **Step 1: Write the failing test**

```python
from app.engine.draft_order import DRAFT_ORDER
from app.pipeline.sources.oracles_elixir import build_champion_id_map, build_game


def _player_row(gid, side, pos, champ, **stats):
    row = {
        "gameid": gid, "datacompleteness": "complete", "date": "2026-11-01 10:00:00",
        "patch": "16.20", "side": side, "position": pos, "teamname": f"{side} Team",
        "teamid": f"oe-{side.lower()}", "champion": champ,
        "kills": "0", "assists": "0", "teamkills": "10", "damageshare": "0.2",
        "totaldamagetaken": "15000", "visionscore": "30", "earnedgoldshare": "0.2",
        "result": "1" if side == "Blue" else "0",
    }
    row.update({key: str(value) for key, value in stats.items()})
    return row


def _team_row(gid, side, bans, result):
    row = {
        "gameid": gid, "datacompleteness": "complete", "date": "2026-11-01 10:00:00",
        "patch": "16.20", "side": side, "position": "team", "teamname": f"{side} Team",
        "teamid": f"oe-{side.lower()}", "champion": "", "result": result,
        "kills": "10", "assists": "0", "teamkills": "10", "damageshare": "0",
        "totaldamagetaken": "0", "visionscore": "0", "earnedgoldshare": "0",
    }
    for index, ban in enumerate(bans, start=1):
        row[f"ban{index}"] = ban
    return row


def _complete_game(gid="g1"):
    positions = ("top", "jng", "mid", "bot", "sup")
    rows = []
    for side in ("Blue", "Red"):
        for pos in positions:
            rows.append(_player_row(gid, side, pos, f"{side}{pos}"))
    rows.append(_team_row(gid, "Blue", ["Bb1", "Bb2", "Bb3", "Bb4", "Bb5"], "1"))
    rows.append(_team_row(gid, "Red", ["Rb1", "Rb2", "Rb3", "Rb4", "Rb5"], "0"))
    return rows


def test_build_game_emits_full_draft_order_aligned_sequence() -> None:
    game = build_game(_complete_game(), build_champion_id_map({}))

    assert game is not None
    assert game.winner_team_id == game.blue_team_id == "oe-blue"
    assert game.red_team_id == "oe-red"
    assert len(game.actions) == len(DRAFT_ORDER)
    for slot, action in zip(DRAFT_ORDER, game.actions, strict=True):
        assert action.action_type == slot.action
        expected_team = game.blue_team_id if slot.side == "blue" else game.red_team_id
        assert action.team_id == expected_team
    assert len(game.lineups) == 10
    assert {line.role for line in game.lineups} == {"top", "jungle", "mid", "bot", "support"}


def test_build_game_skips_incomplete_draft() -> None:
    rows = _complete_game()
    rows = [row for row in rows if not (row["position"] == "sup" and row["side"] == "Blue")]
    assert build_game(rows, build_champion_id_map({})) is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && uv run pytest tests/pipeline/test_oracles_elixir.py -q`
Expected: FAIL with `ImportError: cannot import name 'build_game'`.

- [ ] **Step 3: Implement `build_game`**

Add to `oracles_elixir.py` (imports at top: `from app.engine.draft_order import DRAFT_ORDER`, `from app.pipeline.dataset import GameDraft`, `from app.pipeline.drafts import DraftAction, PlayerLine, Role`):

```python
_POSITION_ORDER = ("top", "jng", "mid", "bot", "sup")


def _team_id(row: Mapping[str, str]) -> str:
    return row.get("teamid", "").strip() or normalize_champion_name(row.get("teamname", ""))


def build_game(
    rows: list[Mapping[str, str]], name_to_id: Mapping[str, str]
) -> GameDraft | None:
    """Build one OE game into a GameDraft, or None if its draft is incomplete."""
    team_rows = {
        row["side"].strip().capitalize(): row
        for row in rows
        if row.get("position", "").strip().lower() == "team"
    }
    if set(team_rows) != {"Blue", "Red"}:
        return None

    players: dict[str, list[Mapping[str, str]]] = {"Blue": [], "Red": []}
    for row in rows:
        position = row.get("position", "").strip().lower()
        side = row.get("side", "").strip().capitalize()
        if position in OE_POSITION_ROLES and side in players:
            players[side].append(row)

    picks: dict[str, list[Mapping[str, str]]] = {}
    bans: dict[str, list[str]] = {}
    for side in ("Blue", "Red"):
        ordered = sorted(
            players[side],
            key=lambda row: _POSITION_ORDER.index(row["position"].strip().lower()),
        )
        side_bans = [
            team_rows[side].get(f"ban{index}", "").strip() for index in range(1, 6)
        ]
        if len(ordered) != 5 or any(not ban for ban in side_bans):
            return None
        picks[side] = ordered
        bans[side] = side_bans

    team_ids = {side: _team_id(team_rows[side]) for side in ("Blue", "Red")}
    winner_side = next(
        (side for side in ("Blue", "Red") if team_rows[side].get("result", "").strip() == "1"),
        None,
    )

    pick_cursor = {"Blue": 0, "Red": 0}
    ban_cursor = {"Blue": 0, "Red": 0}
    actions: list[DraftAction] = []
    for slot in DRAFT_ORDER:
        side = "Blue" if slot.side == "blue" else "Red"
        if slot.action == "ban":
            name = bans[side][ban_cursor[side]]
            ban_cursor[side] += 1
        else:
            name = picks[side][pick_cursor[side]]["champion"].strip()
            pick_cursor[side] += 1
        actions.append(
            DraftAction(
                sequence_number=slot.index + 1,
                action_type=slot.action,
                team_id=team_ids[side],
                champion_id=resolve_champion_id(name, name_to_id),
                champion_name=name,
            )
        )

    lineups: list[PlayerLine] = []
    for side in ("Blue", "Red"):
        for row in picks[side]:
            team_kills = _number(row.get("teamkills"))
            participations = _number(row.get("kills")) + _number(row.get("assists"))
            role: Role = OE_POSITION_ROLES[row["position"].strip().lower()]
            lineups.append(
                PlayerLine(
                    team_id=team_ids[side],
                    role=role,
                    champion_id=resolve_champion_id(row["champion"].strip(), name_to_id),
                    damage_share=_number(row.get("damageshare")),
                    damage_taken=_number(row.get("totaldamagetaken")),
                    vision_score=_number(row.get("visionscore")),
                    gold_share=_number(row.get("earnedgoldshare")),
                    kill_participation=participations / team_kills if team_kills else 0.0,
                )
            )

    gameid = rows[0]["gameid"]
    return GameDraft(
        game_id=gameid,
        series_id=gameid,
        game_number=1,
        patch=(team_rows["Blue"].get("patch", "").strip() or "unknown"),
        finished=True,
        winner_team_id=team_ids[winner_side] if winner_side else None,
        blue_team_id=team_ids["Blue"],
        red_team_id=team_ids["Red"],
        actions=actions,
        started_at=team_rows["Blue"].get("date", "").strip(),
        lineups=lineups,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && uv run pytest tests/pipeline/test_oracles_elixir.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add app/pipeline/sources/oracles_elixir.py tests/pipeline/test_oracles_elixir.py
git commit -m "feat: build Oracle's Elixir games into draft-order-aligned GameDrafts"
```

---

### Task 3: Parse a whole CSV into GameDrafts

**Files:**
- Modify: `backend/app/pipeline/sources/oracles_elixir.py`
- Test: `backend/tests/pipeline/test_oracles_elixir.py`

**Interfaces:**
- Produces:
  - `parse_oracles_elixir(rows: Iterable[Mapping[str, str]], name_to_id: Mapping[str, str]) -> list[GameDraft]`
  - `load_oracles_elixir(csv_path: Path, champions: Mapping[str, Champion]) -> list[GameDraft]`

**Behavior:** Group rows by `gameid` preserving first-seen order. Skip games whose `datacompleteness` (any row) is not `complete`. Delegate each group to `build_game`; drop `None` results. `load_oracles_elixir` reads the CSV with `csv.DictReader` (`encoding="utf-8-sig"`) and builds the name map from `champions`.

- [ ] **Step 1: Write the failing test**

```python
import csv
from pathlib import Path

from app.pipeline.meta import Champion
from app.pipeline.sources.oracles_elixir import load_oracles_elixir, parse_oracles_elixir


def test_parse_keeps_complete_games_and_drops_incomplete() -> None:
    good = _complete_game("good")
    partial = [row for row in _complete_game("partial") if row["position"] != "sup"]
    flagged = _complete_game("flagged")
    for row in flagged:
        row["datacompleteness"] = "partial"

    games = parse_oracles_elixir([*good, *partial, *flagged], {})

    assert [game.game_id for game in games] == ["good"]


def test_load_reads_csv_and_resolves_champions(tmp_path: Path) -> None:
    rows = _complete_game("g1")
    csv_path = tmp_path / "oe.csv"
    fieldnames = sorted({key for row in rows for key in row})
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    champions = {"Bluetop": Champion(character_id="grid-x", name="Bluetop", games_played=1)}

    games = load_oracles_elixir(csv_path, champions)

    assert len(games) == 1
    assert games[0].actions[DRAFT_ORDER_PICK_INDEX].champion_id == "grid-x"
```

Add at the top of the test module: `from app.engine.draft_order import DRAFT_ORDER` and a module constant
`DRAFT_ORDER_PICK_INDEX = next(i for i, slot in enumerate(DRAFT_ORDER) if slot.action == "pick" and slot.side == "blue")`
(the first blue pick slot, whose champion is blue's `top` pick `"Bluetop"`).

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && uv run pytest tests/pipeline/test_oracles_elixir.py -q`
Expected: FAIL with `ImportError: cannot import name 'parse_oracles_elixir'`.

- [ ] **Step 3: Implement the parsers**

Add imports (`import csv`, `from collections import OrderedDict`, `from collections.abc import Iterable`, `from pathlib import Path`) and:

```python
def parse_oracles_elixir(
    rows: Iterable[Mapping[str, str]], name_to_id: Mapping[str, str]
) -> list[GameDraft]:
    """Group OE rows by game and build every complete game into a GameDraft."""
    grouped: "OrderedDict[str, list[Mapping[str, str]]]" = OrderedDict()
    for row in rows:
        gameid = row.get("gameid", "").strip()
        if gameid:
            grouped.setdefault(gameid, []).append(row)

    games: list[GameDraft] = []
    for group in grouped.values():
        if any(row.get("datacompleteness", "").strip().lower() != "complete" for row in group):
            continue
        game = build_game(group, name_to_id)
        if game is not None:
            games.append(game)
    return games


def load_oracles_elixir(
    csv_path: Path, champions: Mapping[str, Champion]
) -> list[GameDraft]:
    """Read an Oracle's Elixir CSV into GameDrafts using GRID champion ids."""
    name_to_id = build_champion_id_map(champions)
    with csv_path.open(newline="", encoding="utf-8-sig") as file:
        return parse_oracles_elixir(list(csv.DictReader(file)), name_to_id)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && uv run pytest tests/pipeline/test_oracles_elixir.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add app/pipeline/sources/oracles_elixir.py tests/pipeline/test_oracles_elixir.py
git commit -m "feat: parse Oracle's Elixir CSV into GameDrafts, skipping incomplete games"
```

---

### Task 4: Build CLI merge + mixed-source training smoke test

**Files:**
- Modify: `backend/app/pipeline/sources/oracles_elixir.py`
- Test: `backend/tests/pipeline/test_oracles_elixir.py`
- Test: `backend/tests/model/test_training.py`

**Interfaces:**
- Produces: `main(argv: list[str] | None = None) -> None` — reads `--csv`, loads GRID champions from `--data-dir`, appends OE games to `--out/drafts.json` (creating it if absent), prints counts and the number of unmatched champion names.

**Behavior:** Load existing `drafts.json` games (if present) via `GameDraft.model_validate`, append OE games, de-duplicate by `game_id` (OE ids are prefixed with `oe:` to avoid colliding with GRID ids — see Step 3), write the merged list back. Log the count of champion names that fell back (unmatched) as a known limitation.

- [ ] **Step 1: Write the failing CLI test**

```python
import json

from app.pipeline.dataset import GameDraft
from app.pipeline.sources.oracles_elixir import main


def test_main_appends_oe_games_to_existing_drafts(tmp_path, capsys, monkeypatch) -> None:
    rows = _complete_game("g1")
    csv_path = tmp_path / "oe.csv"
    fieldnames = sorted({key for row in rows for key in row})
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    out = tmp_path / "derived"
    out.mkdir()
    (out / "drafts.json").write_text("[]", encoding="utf-8")

    monkeypatch.setattr(
        "app.pipeline.sources.oracles_elixir.load_champions", lambda _data_dir: {}
    )
    main(["--csv", str(csv_path), "--data-dir", str(tmp_path), "--out", str(out)])

    payload = json.loads((out / "drafts.json").read_text(encoding="utf-8"))
    games = [GameDraft.model_validate(item) for item in payload]
    assert [game.game_id for game in games] == ["oe:g1"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && uv run pytest tests/pipeline/test_oracles_elixir.py::test_main_appends_oe_games_to_existing_drafts -q`
Expected: FAIL with `ImportError: cannot import name 'main'`.

- [ ] **Step 3: Implement `main` with `oe:`-prefixed ids**

In `build_game`, prefix the OE game id to avoid colliding with GRID ids: set `gameid = "oe:" + rows[0]["gameid"].strip()` and use it for both `game_id` and `series_id`. (Update the Task 2 return accordingly; the Task 3 CSV test's `DRAFT_ORDER_PICK_INDEX` assertion is unaffected.)

Add the CLI (imports: `import argparse`, `from app.pipeline.meta import load_champions`):

```python
def main(argv: list[str] | None = None) -> None:
    """Merge an Oracle's Elixir CSV into the local drafts.json training set."""
    parser = argparse.ArgumentParser(description="Add Oracle's Elixir games to drafts.json.")
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    arguments = parser.parse_args(argv)

    champions = load_champions(arguments.data_dir)
    oe_games = load_oracles_elixir(arguments.csv, champions)

    drafts_path = arguments.out / "drafts.json"
    existing: list[GameDraft] = []
    if drafts_path.is_file():
        existing = [
            GameDraft.model_validate(item)
            for item in json.loads(drafts_path.read_text(encoding="utf-8"))
        ]
    by_id = {game.game_id: game for game in existing}
    for game in oe_games:
        by_id[game.game_id] = game
    merged = list(by_id.values())
    drafts_path.write_text(
        json.dumps([game.model_dump(mode="json") for game in merged], indent=0) + "\n",
        encoding="utf-8",
    )
    print(f"Added {len(oe_games)} Oracle's Elixir games; drafts.json now has {len(merged)}.")
```

Add `import json` at the top.

- [ ] **Step 4: Write the mixed-source training smoke test**

Add to `backend/tests/model/test_training.py`:

```python
def test_train_model_runs_on_mixed_grid_and_oe_synthetic_games(
    make_game: Any, tmp_path: Path
) -> None:
    from app.pipeline.sources.oracles_elixir import build_game, build_champion_id_map

    grid = [_planted_game(make_game, index, index % 2 == 0) for index in range(12)]
    oe = []
    for index in range(12):
        rows = _oe_complete_game(f"m{index}", blue_wins=index % 2 == 0)  # helper in test file
        game = build_game(rows, build_champion_id_map({}))
        assert game is not None
        oe.append(game)

    report = train_model([*grid, *oe], tmp_path, epochs=2, hidden_size=8, seed=11, device="cpu")

    assert report.metrics["model"]["auc"] is not None
```

Add an `_oe_complete_game(gameid, *, blue_wins)` helper mirroring Task 2's `_complete_game` but with the winning side's `result` set from `blue_wins` and distinct champion names per game so the vocabulary is non-trivial. (Copy the row builders into `test_training.py` or import them — prefer a small local helper to keep the model tests self-contained.)

- [ ] **Step 5: Run full gate**

Run: `cd backend && uv run ruff check . && uv run ruff format --check . && uv run basedpyright && uv run pytest -q`
Expected: all green.

- [ ] **Step 6: Commit**

```bash
git add app/pipeline/sources/oracles_elixir.py tests/pipeline/test_oracles_elixir.py tests/model/test_training.py
git commit -m "feat: merge Oracle's Elixir games into drafts.json and train across sources"
```

---

### Task 5: Retrain, measure, and open the PR

Not a code task — run on the machine with real data.

- [ ] Download an Oracle's Elixir match CSV locally (never commit it).
- [ ] `uv run python -m app.pipeline.build` to refresh GRID `drafts.json`, then
      `uv run python -m app.pipeline.sources.oracles_elixir --csv <oe.csv> --data-dir <grid> --out .derived`.
- [ ] Retrain (`uv run python -m app.model.train`) and, optionally, re-run `uv run python -m app.model.diagnose` to see whether the added OE games raise the random-split ceiling.
- [ ] Record before/after accuracy / AUC / log_loss vs. the Milestone 1 result in the PR description (the metrics file is gitignored).
- [ ] Open the PR with `Closes #43`; squash-merge per `CONVENTIONS.md`.

## Self-Review

- **Spec coverage:** new `app/pipeline/sources/oracles_elixir.py` parser (Tasks 1–3) ✓; normalizes into unchanged `GameDraft`/`DraftAction`/`PlayerLine` ✓; champion name→GRID id with logged fallback (Task 1) ✓; synthesized pick order via `DRAFT_ORDER` (Task 2) ✓; missing fields → 0 (`_number`, Task 1) ✓; manual build step into `.derived/` (Task 4) ✓; synthetic fixtures only ✓; mixed-source `train_model` smoke test (Task 4) ✓; retrain + before/after AUC (Task 5) ✓.
- **Draft-order alignment:** Task 2 emits the full 20-slot sequence in `DRAFT_ORDER` order and returns `None` for incomplete drafts, so `actions[i]` ↔ `DRAFT_ORDER[i]` holds for OE exactly as for GRID. ✓
- **Type consistency:** `build_game` and `parse_oracles_elixir` return `GameDraft`/`list[GameDraft]`; `resolve_champion_id`/`_number` signatures are stable across tasks. ✓
- **Known limitation to document in the PR:** OE team ids are source-local (`oe:`/normalized name), so cross-source team-tendency merging is best-effort only, per the spec's dedup note.
