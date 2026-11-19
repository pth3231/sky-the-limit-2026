"""Typed loaders for the GRID metadata CSV files."""

import csv
from collections.abc import Iterator
from pathlib import Path

from pydantic import BaseModel, ConfigDict, TypeAdapter

_BOOL_ADAPTER = TypeAdapter(bool)


class GameMeta(BaseModel):
    """Game metadata needed to contextualize a draft."""

    model_config = ConfigDict(frozen=True)

    game_id: str
    series_id: str
    game_number: int
    patch: str
    finished: bool
    winner_team_id: str | None
    blue_team_id: str
    red_team_id: str


class Champion(BaseModel):
    """Champion metadata used to label draft actions."""

    model_config = ConfigDict(frozen=True)

    character_id: str
    name: str
    games_played: int


class Team(BaseModel):
    """Team metadata used to identify historical draft participants."""

    model_config = ConfigDict(frozen=True)

    team_id: str
    name: str


def _lol_rows(data_dir: Path, filename: str) -> Iterator[dict[str, str]]:
    path = data_dir / "meta" / filename
    with path.open(newline="", encoding="utf-8-sig") as file:
        for row in csv.DictReader(file):
            if row.get("title", "").strip().casefold() == "lol":
                yield row


def _game_from_row(row: dict[str, str]) -> GameMeta:
    sides = {
        row["team1_side"].strip().casefold(): row["team1_id"],
        row["team2_side"].strip().casefold(): row["team2_id"],
    }
    if set(sides) != {"blue", "red"}:
        raise ValueError(f"Game {row['game_id']} must have one blue and one red team")

    winner_team_id = row["winner_team_id"].strip() or None
    return GameMeta(
        game_id=row["game_id"],
        series_id=row["series_id"],
        game_number=int(row["game_number"]),
        patch=row["patch"],
        finished=_BOOL_ADAPTER.validate_python(row["finished"]),
        winner_team_id=winner_team_id,
        blue_team_id=sides["blue"],
        red_team_id=sides["red"],
    )


def load_games(data_dir: Path) -> dict[str, GameMeta]:
    """Load LoL game metadata indexed by GRID game ID."""
    return {
        game.game_id: game
        for game in (_game_from_row(row) for row in _lol_rows(data_dir, "games.csv"))
    }


def load_champions(data_dir: Path) -> dict[str, Champion]:
    """Load LoL champion metadata indexed by GRID character ID."""
    champions = (
        Champion(
            character_id=row["character_id"],
            name=row["name"],
            games_played=int(row["games_played"]),
        )
        for row in _lol_rows(data_dir, "characters.csv")
    )
    return {champion.character_id: champion for champion in champions}


def load_teams(data_dir: Path) -> dict[str, Team]:
    """Load LoL team metadata indexed by GRID team ID."""
    teams = (
        Team(team_id=row["team_id"], name=row["name"]) for row in _lol_rows(data_dir, "teams.csv")
    )
    return {team.team_id: team for team in teams}
