import csv
from pathlib import Path

import pytest


def write_rows(root: Path, filename: str, rows: list[dict[str, str]]) -> None:
    meta = root / "meta"
    meta.mkdir(exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with (meta / filename).open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def game(game_id: str, date: str, **changes: str) -> dict[str, str]:
    return {
        "title": "lol",
        "series_id": f"s-{game_id}",
        "game_id": game_id,
        "game_number": "1",
        "started_at": date,
        "team1_id": "c9",
        "team2_id": "opp",
        **changes,
    }


def roster(game_id: str, player: str, **changes: str) -> dict[str, str]:
    return {
        "title": "lol",
        "series_id": f"s-{game_id}",
        "game_id": game_id,
        "game_number": "1",
        "team_id": "c9",
        "team_name": "Cloud9",
        "side": "blue",
        "player_id": player,
        "player_name": f"Row {player}",
        "character_id": "champ",
        "character_name": "Champion",
        "participation_status": "active",
        **changes,
    }


def player(player_id: str, name: str, **changes: str) -> dict[str, str]:
    return {"title": "lol", "player_id": player_id, "name": name, **changes}


def load(root: Path):
    from app.pipeline.rosters import load_rosters

    return load_rosters(root)


def test_only_active_participants_are_included(tmp_path: Path) -> None:
    write_rows(tmp_path, "games.csv", [game("g", "2026-01-01T00:00:00Z")])
    write_rows(
        tmp_path,
        "players.csv",
        [player("active", "Active"), player("inactive", "Inactive"), player("played", "Played")],
    )
    write_rows(
        tmp_path,
        "rosters.csv",
        [
            roster("g", "active", participation_status="active"),
            roster("g", "inactive", participation_status="inactive"),
            roster("g", "played", participation_status="played"),
        ],
    )

    assert [p.player_id for p in load(tmp_path)["c9"].players] == ["active"]


@pytest.mark.parametrize("with_older_snapshot", [False, True])
def test_overfull_snapshot_is_rejected_without_truncating_players(
    tmp_path: Path, with_older_snapshot: bool
) -> None:
    games = [game("new", "2026-02-01T00:00:00Z")]
    players = [player(f"p{i}", f"Player {i}") for i in range(6)]
    players += [player(f"o{i}", f"Opponent {i}") for i in range(5)]
    rows = [roster("new", f"p{i}") for i in range(6)]
    rows += [roster("new", f"o{i}", team_id="opp") for i in range(5)]
    rows.append(roster("new", "o0", team_id="opp"))
    if with_older_snapshot:
        games.append(game("old", "2026-01-01T00:00:00Z"))
        rows += [roster("old", f"p{i}") for i in range(5)]

    write_rows(tmp_path, "games.csv", games)
    write_rows(tmp_path, "players.csv", players)
    write_rows(tmp_path, "rosters.csv", rows)

    result = load(tmp_path)

    if with_older_snapshot:
        assert result["c9"].as_of == "2026-01-01T00:00:00Z"
        assert [p.player_id for p in result["c9"].players] == [f"p{i}" for i in range(5)]
    else:
        assert "c9" not in result
    assert result["opp"].as_of == "2026-02-01T00:00:00Z"
    assert [p.player_id for p in result["opp"].players] == [f"o{i}" for i in range(5)]


def test_conflicting_team_membership_does_not_hide_overfull_snapshot(tmp_path: Path) -> None:
    write_rows(tmp_path, "games.csv", [game("g", "2026-01-01T00:00:00Z")])
    write_rows(tmp_path, "players.csv", [player(f"p{i}", f"Player {i}") for i in range(6)])
    write_rows(
        tmp_path,
        "rosters.csv",
        [roster("g", f"p{i}") for i in range(6)] + [roster("g", "p5", team_id="opp")],
    )

    assert "c9" not in load(tmp_path)


def test_latest_snapshot_uses_verified_membership_and_explicit_roles(tmp_path: Path) -> None:
    write_rows(
        tmp_path,
        "games.csv",
        [
            game("old", "2026-01-01T12:00:00Z"),
            game("new", "2026-02-01T12:00:00Z"),
            game("unknown", ""),
            game("val", "2027-01-01T00:00:00Z", title="val"),
        ],
    )
    write_rows(
        tmp_path,
        "players.csv",
        [
            player("p1", "Alice"),
            player("p2", "Bob"),
            player("p3", "Carol"),
            player("p4", "Dana"),
            player("foreign", "Other", title="val"),
        ],
    )
    write_rows(
        tmp_path,
        "rosters.csv",
        [
            roster("old", "p3"),
            roster("unknown", "p3"),
            roster("val", "p3"),
            roster("new", "p2"),
            roster("new", "p1", position="middle"),
            roster("new", "p3", participation_status="bench"),
            roster("new", "missing"),
            roster("new", "foreign"),
            roster("new", ""),
            roster("new", "p4", series_id="wrong"),
            roster("new", "p4", game_number="2"),
            roster("new", "p4", team_id="other"),
            roster("new", "p3", team_id="opp", position="support"),
        ],
    )

    result = load(tmp_path)

    assert result["c9"].as_of == "2026-02-01T12:00:00Z"
    assert [(p.player_id, p.name, p.role) for p in result["c9"].players] == [
        ("p1", "Alice", "mid"),
        ("p2", "Bob", None),
    ]
    assert [(p.player_id, p.role) for p in result["opp"].players] == [("p3", "support")]


def test_invalid_dates_and_conflicting_games_do_not_create_snapshots(tmp_path: Path) -> None:
    write_rows(
        tmp_path,
        "games.csv",
        [
            game("unknown", "not-a-date"),
            game("bad", "2027-01-01T00:00:00Z"),
            game("bad", "2027-01-01T00:00:00Z", team2_id="other"),
            game("good", "2026-01-01T00:00:00Z"),
        ],
    )
    write_rows(tmp_path, "players.csv", [player("p", "Player")])
    write_rows(tmp_path, "rosters.csv", [roster(x, "p") for x in ("unknown", "bad", "good")])

    result = load(tmp_path)

    assert result["c9"].as_of == "2026-01-01T00:00:00Z"
    assert [p.player_id for p in result["c9"].players] == ["p"]


def test_ambiguous_players_and_roles_are_not_invented(tmp_path: Path) -> None:
    write_rows(tmp_path, "games.csv", [game("g", "2026-01-01T00:00:00Z")])
    write_rows(
        tmp_path,
        "players.csv",
        [
            player("p1", "Alice"),
            player("p2", "Bob"),
            player("p3", "Carol"),
            player("p4", "Name"),
            player("p4", "Different"),
            player("p5", "Eve"),
        ],
    )
    write_rows(
        tmp_path,
        "rosters.csv",
        [
            roster("g", "p1", position="top"),
            roster("g", "p1", position="mid"),
            roster("g", "p2", position="support"),
            roster("g", "p3", position="support"),
            roster("g", "p4"),
            roster("g", "p5"),
            roster("g", "p5", team_id="opp"),
        ],
    )

    result = load(tmp_path)

    assert [(p.player_id, p.role) for p in result["c9"].players] == [
        ("p1", None),
        ("p2", None),
        ("p3", None),
    ]
    assert "opp" not in result


def test_same_timestamp_tie_and_output_are_independent_of_row_order(tmp_path: Path) -> None:
    games = [game("a", "2026-01-01T00:00:00Z"), game("b", "2026-01-01T00:00:00Z")]
    players = [player("p1", "Alice"), player("p2", "Bob")]
    rows = [roster("a", "p1"), roster("b", "p2")]
    for filename, records in (
        ("games.csv", games),
        ("players.csv", players),
        ("rosters.csv", rows),
    ):
        write_rows(tmp_path, filename, records)
    first = load(tmp_path)
    for filename, records in (
        ("games.csv", games),
        ("players.csv", players),
        ("rosters.csv", rows),
    ):
        write_rows(tmp_path, filename, list(reversed(records)))
    assert first == load(tmp_path)
    assert [p.player_id for p in first["c9"].players] == ["p2"]


def test_equivalent_explicit_position_names_are_not_conflicting(tmp_path: Path) -> None:
    write_rows(tmp_path, "games.csv", [game("g", "2026-01-01T00:00:00Z")])
    write_rows(tmp_path, "players.csv", [player("p", "Player")])
    write_rows(
        tmp_path,
        "rosters.csv",
        [
            roster("g", "p", position="mid"),
            roster("g", "p", position="middle"),
        ],
    )

    assert load(tmp_path)["c9"].players[0].role == "mid"


def test_timezones_determine_latest_snapshot_not_timestamp_text(tmp_path: Path) -> None:
    write_rows(
        tmp_path,
        "games.csv",
        [
            game("earlier", "2026-02-01T02:00:00+03:00"),
            game("later", "2026-02-01T00:00:00Z"),
        ],
    )
    write_rows(tmp_path, "players.csv", [player("p1", "Earlier"), player("p2", "Later")])
    write_rows(tmp_path, "rosters.csv", [roster("earlier", "p1"), roster("later", "p2")])

    assert [p.player_id for p in load(tmp_path)["c9"].players] == ["p2"]


def test_no_dated_snapshot_means_no_membership(tmp_path: Path) -> None:
    write_rows(tmp_path, "games.csv", [game("g", "")])
    write_rows(tmp_path, "players.csv", [player("p", "Player")])
    write_rows(tmp_path, "rosters.csv", [roster("g", "p")])

    assert load(tmp_path) == {}


@pytest.mark.parametrize("missing", ["games.csv", "players.csv", "rosters.csv"])
def test_missing_metadata_returns_empty_snapshots(tmp_path: Path, missing: str) -> None:
    records = {
        "games.csv": [game("g", "2026-01-01T00:00:00Z")],
        "players.csv": [player("p", "Player")],
        "rosters.csv": [roster("g", "p")],
    }
    for filename, rows in records.items():
        if filename != missing:
            write_rows(tmp_path, filename, rows)
    assert load(tmp_path) == {}


def test_real_rosters_csv_schema_without_position_column(tmp_path: Path) -> None:
    """Test with the exact column set from the real GRID rosters.csv (no position column)."""
    meta = tmp_path / "meta"
    meta.mkdir(exist_ok=True)
    # Games with real schema
    with (meta / "games.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "title",
                "series_id",
                "game_id",
                "game_number",
                "started_at",
                "team1_id",
                "team2_id",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "title": "lol",
                "series_id": "s-g",
                "game_id": "g",
                "game_number": "1",
                "started_at": "2026-01-01T00:00:00Z",
                "team1_id": "c9",
                "team2_id": "opp",
            }
        )
    # Players with real schema
    with (meta / "players.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["title", "player_id", "name"])
        writer.writeheader()
        for i in range(5):
            writer.writerow({"title": "lol", "player_id": f"p{i}", "name": f"Player{i}"})
    # Rosters with REAL schema - NO position column
    with (meta / "rosters.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "title",
                "series_id",
                "game_id",
                "game_number",
                "team_id",
                "team_name",
                "side",
                "player_id",
                "player_name",
                "character_id",
                "character_name",
                "participation_status",
            ],
        )
        writer.writeheader()
        for i in range(5):
            writer.writerow(
                {
                    "title": "lol",
                    "series_id": "s-g",
                    "game_id": "g",
                    "game_number": "1",
                    "team_id": "c9",
                    "team_name": "Cloud9",
                    "side": "blue",
                    "player_id": f"p{i}",
                    "player_name": f"Player{i}",
                    "character_id": f"champ{i}",
                    "character_name": f"Champion{i}",
                    "participation_status": "active",
                }
            )

    result = load(tmp_path)

    assert "c9" in result
    assert len(result["c9"].players) == 5
    # All roles should be None since there's no position column
    assert all(p.role is None for p in result["c9"].players)
    assert [p.player_id for p in result["c9"].players] == ["p0", "p1", "p2", "p3", "p4"]
