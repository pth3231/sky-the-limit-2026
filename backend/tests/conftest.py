import json
from collections.abc import Callable, Sequence
from itertools import count
from pathlib import Path
from typing import Any, Literal

import pytest
import zstandard

from app.engine.service import DraftEngine
from app.pipeline.aggregates import compute_aggregates
from app.pipeline.dataset import GameDraft
from app.pipeline.drafts import DraftAction
from app.pipeline.meta import Champion, Team


@pytest.fixture
def write_zstd_feed(tmp_path: Path) -> Callable[[str, list[dict[str, Any]]], Path]:
    def write_feed(filename: str, envelopes: list[dict[str, Any]]) -> Path:
        path = tmp_path / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        with zstandard.open(path, "wt", encoding="utf-8") as file:
            for envelope in envelopes:
                file.write(json.dumps(envelope) + "\n")
        return path

    return write_feed


@pytest.fixture
def make_game() -> Callable[..., GameDraft]:
    game_numbers = count(1)

    def make(
        patch: str = "16.1",
        *,
        winner: Literal["blue", "red"] | None = "blue",
        blue: str = "c9",
        red: str = "opp",
        blue_picks: Sequence[str] = (),
        red_picks: Sequence[str] = (),
        blue_bans: Sequence[str] = (),
        red_bans: Sequence[str] = (),
    ) -> GameDraft:
        entries: list[tuple[Literal["ban", "pick"], str, str]] = [
            *(("ban", blue, champion) for champion in blue_bans),
            *(("ban", red, champion) for champion in red_bans),
            *(("pick", blue, champion) for champion in blue_picks),
            *(("pick", red, champion) for champion in red_picks),
        ]
        actions = [
            DraftAction(
                sequence_number=sequence,
                action_type=kind,
                team_id=team_id,
                champion_id=champion,
                champion_name=champion,
            )
            for sequence, (kind, team_id, champion) in enumerate(entries, start=1)
        ]
        game_id = f"game-{next(game_numbers)}"
        return GameDraft(
            game_id=game_id,
            series_id=f"series-{game_id}",
            game_number=1,
            patch=patch,
            finished=True,
            winner_team_id={"blue": blue, "red": red}[winner] if winner else None,
            blue_team_id=blue,
            red_team_id=red,
            actions=actions,
        )

    return make


CHAMPION_NAMES = {f"c{number}": f"Champ {number}" for number in range(24)}
TEAM_NAMES = {
    "c9": "Cloud9 Kia",
    "opp": "Rivals",
    "t1": "Team One",
    "t2": "Team Two",
    "ghost": "Ghost Team",
}


@pytest.fixture
def synthetic_drafts(make_game: Callable[..., GameDraft]) -> list[GameDraft]:
    matchup = {"blue": "c9", "red": "opp", "blue_picks": ["c0", "c1"], "red_picks": ["c2", "c3"]}
    return [
        *(make_game(winner="blue", red_bans=["c4"], **matchup) for _ in range(12)),
        *(make_game(winner="red", red_bans=["c4"], **matchup) for _ in range(8)),
        *(make_game(blue="t1", red="t2", winner="blue", blue_picks=["c5"]) for _ in range(15)),
        *(make_game(blue="t1", red="t2", winner="red", red_picks=["c6"]) for _ in range(5)),
    ]


@pytest.fixture
def engine(synthetic_drafts: list[GameDraft]) -> DraftEngine:
    champions = {
        champion_id: Champion(character_id=champion_id, name=name, games_played=1)
        for champion_id, name in CHAMPION_NAMES.items()
    }
    teams = {team_id: Team(team_id=team_id, name=name) for team_id, name in TEAM_NAMES.items()}
    return DraftEngine(
        compute_aggregates(synthetic_drafts),
        champions,
        teams,
        our_team_id="c9",
        min_games=1,
    )
