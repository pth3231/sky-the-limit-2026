"""Load verified historical team rosters from local metadata."""

import csv
from collections import Counter, defaultdict
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from app.engine.models import RosterPlayer
from app.pipeline.meta import _ROLE_NAMES, Role


class RosterSnapshot(BaseModel):
    """A dated roster observed in one historical game."""

    model_config = ConfigDict(frozen=True)

    as_of: str
    players: list[RosterPlayer]


@dataclass(frozen=True)
class _Game:
    series_id: str
    game_id: str
    number: int
    started_at: str
    timestamp: datetime
    teams: tuple[str, str]


def _rows(data_dir: Path, filename: str) -> Iterator[dict[str, str]]:
    with (data_dir / "meta" / filename).open(encoding="utf-8-sig", newline="") as file:
        for row in csv.DictReader(file):
            if (row.get("title") or "").strip().casefold() == "lol":
                yield row


def _value(row: dict[str, str], key: str) -> str:
    return (row.get(key) or "").strip()


def _games(data_dir: Path) -> dict[str, _Game]:
    candidates: dict[str, set[_Game | None]] = defaultdict(set)
    for row in _rows(data_dir, "games.csv"):
        game_id = _value(row, "game_id")
        if not game_id:
            continue
        series_id, date = _value(row, "series_id"), _value(row, "started_at")
        teams = (_value(row, "team1_id"), _value(row, "team2_id"))
        try:
            number = int(_value(row, "game_number"))
            timestamp = datetime.fromisoformat(date.replace("Z", "+00:00"))
            if timestamp.tzinfo is None:
                timestamp = timestamp.replace(tzinfo=UTC)
            timestamp = timestamp.astimezone(UTC)
            if not series_id or number < 1 or not all(teams) or teams[0] == teams[1]:
                raise ValueError("Incomplete game identity")
            candidates[game_id].add(_Game(series_id, game_id, number, date, timestamp, teams))
        except (ValueError, OverflowError):
            candidates[game_id].add(None)
    result: dict[str, _Game] = {}
    for game_id, games in candidates.items():
        if len(games) == 1:
            game = next(iter(games))
            if game is not None:
                result[game_id] = game
    return result


def _players(data_dir: Path) -> dict[str, str]:
    candidates: dict[str, set[str]] = defaultdict(set)
    for row in _rows(data_dir, "players.csv"):
        player_id, name = _value(row, "player_id"), _value(row, "name")
        if player_id:
            candidates[player_id].add(name)
    return {
        player_id: next(iter(names))
        for player_id, names in candidates.items()
        if len(names) == 1 and "" not in names
    }


def load_rosters(data_dir: Path) -> dict[str, RosterSnapshot]:
    """Return each team's latest dated game with verified active participants, if available."""
    try:
        games = _games(data_dir)
        names = _players(data_dir)
        memberships: dict[tuple[str, str], dict[str, set[str]]] = defaultdict(
            lambda: defaultdict(set)
        )
        for row in _rows(data_dir, "rosters.csv"):
            game = games.get(_value(row, "game_id"))
            player_id, team_id = _value(row, "player_id"), _value(row, "team_id")
            if (
                game is None
                or _value(row, "series_id") != game.series_id
                or _value(row, "game_number") != str(game.number)
                or team_id not in game.teams
                or player_id not in names
                or _value(row, "participation_status").casefold() != "active"
            ):
                continue
            memberships[(game.game_id, player_id)][team_id].add(_value(row, "position").casefold())
    except (OSError, UnicodeError, csv.Error):
        return {}

    participant_counts = Counter(
        (game_id, team_id) for (game_id, _), teams in memberships.items() for team_id in teams
    )
    grouped: dict[tuple[str, str], list[tuple[str, Role | None]]] = defaultdict(list)
    for (game_id, player_id), teams in memberships.items():
        if len(teams) != 1:
            continue
        team_id, positions = next(iter(teams.items()))
        explicit: set[Role | None] = {
            _ROLE_NAMES.get(position) for position in positions if position
        }
        role = next(iter(explicit)) if len(explicit) == 1 else None
        grouped[(game_id, team_id)].append((player_id, role))

    latest: dict[str, tuple[tuple[datetime, str, int, str], RosterSnapshot]] = {}
    for (game_id, team_id), members in grouped.items():
        if participant_counts[(game_id, team_id)] > 5:
            continue
        game = games[game_id]
        role_counts = Counter(role for _, role in members if role is not None)
        players = [
            RosterPlayer(
                player_id=player_id,
                name=names[player_id],
                role=role if role is not None and role_counts[role] == 1 else None,
            )
            for player_id, role in sorted(members)
        ]
        key = (game.timestamp, game.series_id, game.number, game.game_id)
        if team_id not in latest or key > latest[team_id][0]:
            latest[team_id] = (key, RosterSnapshot(as_of=game.started_at, players=players))
    return {team_id: snapshot for team_id, (_, snapshot) in latest.items()}
