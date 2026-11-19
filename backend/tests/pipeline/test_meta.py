import csv
from pathlib import Path

from app.pipeline.meta import load_champions, load_games, load_teams


def _write_csv(directory: Path, filename: str, rows: list[dict[str, str]]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / filename).open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_load_games_maps_sides_and_filters_other_titles(tmp_path: Path) -> None:
    _write_csv(
        tmp_path / "meta",
        "games.csv",
        [
            {
                "title": "lol",
                "series_id": "series-1",
                "game_id": "game-1",
                "game_number": "2",
                "patch": "15.10",
                "finished": "true",
                "winner_team_id": "team-2",
                "team1_id": "team-1",
                "team1_side": "red",
                "team2_id": "team-2",
                "team2_side": "blue",
            },
            {
                "title": "val",
                "series_id": "series-2",
                "game_id": "game-2",
                "game_number": "1",
                "patch": "10.00",
                "finished": "true",
                "winner_team_id": "team-3",
                "team1_id": "team-3",
                "team1_side": "blue",
                "team2_id": "team-4",
                "team2_side": "red",
            },
        ],
    )

    games = load_games(tmp_path)

    assert list(games) == ["game-1"]
    assert games["game-1"].game_number == 2
    assert games["game-1"].blue_team_id == "team-2"
    assert games["game-1"].red_team_id == "team-1"
    assert games["game-1"].finished is True


def test_load_champions_and_teams_as_typed_lol_records(tmp_path: Path) -> None:
    _write_csv(
        tmp_path / "meta",
        "characters.csv",
        [
            {"title": "lol", "character_id": "champ-1", "name": "Aster", "games_played": "12"},
            {"title": "val", "character_id": "agent-1", "name": "Agent", "games_played": "4"},
        ],
    )
    _write_csv(
        tmp_path / "meta",
        "teams.csv",
        [
            {"title": "lol", "team_id": "team-1", "name": "Cloud9 Kia"},
            {"title": "val", "team_id": "team-2", "name": "Other Team"},
        ],
    )

    champions = load_champions(tmp_path)
    teams = load_teams(tmp_path)

    assert list(champions) == ["champ-1"]
    assert champions["champ-1"].games_played == 12
    assert teams["team-1"].name == "Cloud9 Kia"
    assert "team-2" not in teams
