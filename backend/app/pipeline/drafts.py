"""Streaming extraction of structured draft actions from GRID series feeds."""

import json
from pathlib import Path
from typing import Any, Literal

import zstandard
from pydantic import BaseModel, ConfigDict


class DraftAction(BaseModel):
    """One champion pick or ban recorded in a game draft."""

    model_config = ConfigDict(frozen=True)

    sequence_number: int
    action_type: Literal["ban", "pick"]
    team_id: str
    champion_id: str
    champion_name: str


def read_final_series_state(path: Path) -> dict[str, Any]:
    """Read the last event-level seriesState snapshot from a compressed feed."""
    last_state: Any = None
    with zstandard.open(path, "rt", encoding="utf-8") as feed:
        for line in feed:
            envelope = json.loads(line)
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
            DraftAction(
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
