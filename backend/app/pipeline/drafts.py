"""Streaming extraction of structured draft actions from GRID series feeds."""

from pathlib import Path
from typing import Any, Literal

import orjson
import zstandard
from pydantic import BaseModel, ConfigDict

Role = Literal["top", "jungle", "mid", "bot", "support"]
_ROLE_NAMES: dict[str, Role] = {
    "top": "top",
    "jungle": "jungle",
    "mid": "mid",
    "middle": "mid",
    "bot": "bot",
    "bottom": "bot",
    "adc": "bot",
    "support": "support",
}


class DraftAction(BaseModel):
    """One champion pick or ban recorded in a game draft."""

    model_config = ConfigDict(frozen=True)

    sequence_number: int
    action_type: Literal["ban", "pick"]
    team_id: str
    champion_id: str
    champion_name: str


class PlayerLine(BaseModel):
    """A player's champion and normalized in-game performance signals."""

    model_config = ConfigDict(frozen=True)

    team_id: str
    role: Role | None
    champion_id: str
    player_id: str | None = None
    damage_share: float = 0.0
    damage_taken: float = 0.0
    vision_score: float = 0.0
    gold_share: float = 0.0
    kill_participation: float = 0.0


def _number(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def read_final_series_state(path: Path) -> dict[str, Any]:
    """Read the last event-level seriesState snapshot from a compressed feed."""
    last_state: Any = None
    with zstandard.open(path, "rt", encoding="utf-8") as feed:
        for line in feed:
            envelope = orjson.loads(line)
            for event in envelope.get("events", []):
                state = event.get("seriesState")
                if state is not None:
                    last_state = state

    if not isinstance(last_state, dict):
        raise ValueError(f"No seriesState snapshot found in GRID feed: {path}")
    return last_state


def extract_game_drafts(series_state: dict[str, Any]) -> dict[str, list[DraftAction]]:
    """Extract and sequence-order each game's structured pick and ban actions."""
    drafts: dict[str, list[DraftAction]] = {}
    for game in series_state.get("games", []):
        actions = [
            DraftAction.model_construct(
                sequence_number=action["sequenceNumber"],
                action_type=action["type"],
                team_id=action["drafter"]["id"],
                champion_id=action["draftable"]["id"],
                champion_name=action["draftable"]["name"],
            )
            for action in game.get("draftActions", [])
        ]
        actions.sort(key=lambda action: action.sequence_number)
        drafts[game["id"]] = actions
    return drafts


def extract_game_lineups(series_state: dict[str, Any]) -> dict[str, list[PlayerLine]]:
    """Extract complete five-player team lineups from each game's final state."""
    lineups: dict[str, list[PlayerLine]] = {}
    for game in series_state.get("games", []):
        game_id = game.get("id")
        if not isinstance(game_id, str):
            continue
        players_by_team: list[PlayerLine] = []
        for team in game.get("teams", []):
            team_id = team.get("id")
            players = team.get("players", [])
            if not isinstance(team_id, str) or not isinstance(players, list) or len(players) < 5:
                continue

            team_kills = _number(team.get("kills"))
            team_gold = _number(team.get("totalMoneyEarned"))
            team_lines: list[PlayerLine] = []
            for player in players[:5]:
                character = player.get("character", {})
                champion_id = character.get("id") if isinstance(character, dict) else None
                if not isinstance(champion_id, str):
                    team_lines = []
                    break

                position = player.get("position")
                role = (
                    _ROLE_NAMES.get(position.strip().casefold())
                    if isinstance(position, str)
                    else None
                )
                player_gold = _number(player.get("totalMoneyEarned"))
                participations = _number(player.get("kills")) + _number(
                    player.get("killAssistsGiven")
                )
                team_lines.append(
                    PlayerLine.model_construct(
                        team_id=team_id,
                        role=role,
                        champion_id=champion_id,
                        damage_share=_number(player.get("damagePercentage")),
                        damage_taken=_number(player.get("damageTaken")),
                        vision_score=_number(player.get("visionScore")),
                        gold_share=player_gold / team_gold if team_gold else 0.0,
                        kill_participation=(participations / team_kills if team_kills else 0.0),
                    )
                )
            players_by_team.extend(team_lines)

        if players_by_team:
            lineups[game_id] = players_by_team
    return lineups
