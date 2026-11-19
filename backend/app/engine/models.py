"""Request and response models for draft recommendations, camelCase on the wire."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field
from pydantic.alias_generators import to_camel

Side = Literal["blue", "red"]
ActionType = Literal["ban", "pick"]
SlotOwner = Literal["us", "opponent"]
SeriesFormat = Literal["bo1", "bo3", "bo5"]
Signal = Literal["strong", "some", "limited"]


class CamelModel(BaseModel):
    """Frozen base model that reads and writes camelCase JSON and accepts snake_case in Python."""

    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True, frozen=True)


class PreviousGame(CamelModel):
    """Picks and result of an earlier game in the series, from our point of view."""

    our_picks: list[str]
    opponent_picks: list[str]
    result: Literal["win", "loss"]


class DraftRequest(CamelModel):
    """The full current series and draft state sent by the client."""

    opponent_team_id: str
    series_format: SeriesFormat = "bo3"
    patch: str | None = None
    our_side: Side
    previous_games: list[PreviousGame] = Field(default_factory=list)
    actions: list[str] = Field(default_factory=list)


class Slot(CamelModel):
    """A draft slot: its position, side, action, and whether it is ours."""

    index: int
    side: Side
    action: ActionType
    team: SlotOwner


class ContributingFactor(CamelModel):
    """One plain-language reason; ``impact`` approximates its change in win probability."""

    label: str
    impact: float
    signal: Signal


class Candidate(CamelModel):
    """A champion suggested for the next slot, with its score and reasons."""

    champion_id: str
    champion_name: str
    score: float
    reason: str
    contributing_factors: list[ContributingFactor]


class OpponentTendency(CamelModel):
    """How often the opponent has picked or banned a champion on the blended patches."""

    champion_id: str
    champion_name: str
    action: ActionType
    count: int
    rate: float
    label: str


class TeamSummary(CamelModel):
    """A team that can be selected as the opponent."""

    team_id: str
    name: str


class ChampionSummary(CamelModel):
    """A champion that can be drafted."""

    champion_id: str
    name: str


class DraftResponse(CamelModel):
    """Recommendation, win rates, and context for the submitted draft state."""

    game_number: int
    patches_used: list[str]
    next_slot: Slot | None
    recommendation: Candidate | None
    alternatives: list[Candidate]
    win_rate: float
    game_win_rate: float
    opponent_tendencies: list[OpponentTendency]
    contributing_factors: list[ContributingFactor]
    fearless_excluded_ids: list[str]
    unavailable_ids: list[str]
    sample_note: str | None
