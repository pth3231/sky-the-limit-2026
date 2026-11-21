import csv
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pydantic import TypeAdapter

from app.api.main import create_app
from app.engine.service import DraftEngine, load_engine
from app.pipeline.aggregates import PatchAggregate, compute_aggregates
from app.pipeline.dataset import GameDraft


def test_missing_roster_metadata_preserves_team_identities_and_draft(engine: DraftEngine) -> None:
    client = TestClient(create_app(engine))

    response = client.get("/rosters", params={"opponentTeamId": "opp"})

    assert response.status_code == 200
    assert response.json() == {
        "ourTeam": {"teamId": "c9", "name": "Cloud9 Kia", "asOf": None, "players": []},
        "opponentTeam": {"teamId": "opp", "name": "Rivals", "asOf": None, "players": []},
    }
    assert (
        client.post(
            "/draft/recommend", json={"opponentTeamId": "opp", "ourSide": "blue"}
        ).status_code
        == 200
    )


def test_invalid_opponents_are_rejected(engine: DraftEngine) -> None:
    client = TestClient(create_app(engine))
    for opponent in ("unknown", "c9"):
        assert client.get("/rosters", params={"opponentTeamId": opponent}).status_code in (404, 422)
    assert client.get("/rosters").status_code == 422


@pytest.mark.parametrize("missing", [None, "games.csv", "players.csv", "rosters.csv"])
def test_load_engine_connects_metadata_to_rosters_api(
    tmp_path: Path, synthetic_drafts: list[GameDraft], missing: str | None
) -> None:
    meta = tmp_path / "meta"
    meta.mkdir()
    fixtures = {
        "teams.csv": [
            {"title": "lol", "team_id": "c9", "name": "Cloud9"},
            {"title": "lol", "team_id": "opp", "name": "Rivals"},
        ],
        "characters.csv": [
            {"title": "lol", "character_id": "c", "name": "Champ", "games_played": "1"}
        ],
        "games.csv": [
            {
                "title": "lol",
                "series_id": "s",
                "game_id": "g",
                "game_number": "1",
                "started_at": "2026-01-01T00:00:00Z",
                "team1_id": "c9",
                "team2_id": "opp",
            }
        ],
        "players.csv": [{"title": "lol", "player_id": "p", "name": "Player"}],
        "rosters.csv": [
            {
                "title": "lol",
                "series_id": "s",
                "game_id": "g",
                "game_number": "1",
                "team_id": "opp",
                "player_id": "p",
                "participation_status": "active",
            }
        ],
    }
    for filename, rows in fixtures.items():
        if filename == missing:
            continue
        with (meta / filename).open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    aggregates = compute_aggregates(synthetic_drafts)
    (tmp_path / "aggregates.json").write_bytes(
        TypeAdapter(dict[str, PatchAggregate]).dump_json(aggregates)
    )

    client = TestClient(create_app(load_engine(tmp_path, tmp_path, "c9")))
    response = client.get("/rosters", params={"opponentTeamId": "opp"})

    assert response.status_code == 200
    assert response.json()["opponentTeam"] == {
        "teamId": "opp",
        "name": "Rivals",
        "asOf": "2026-01-01T00:00:00Z" if missing is None else None,
        "players": [{"playerId": "p", "name": "Player", "role": None}] if missing is None else [],
    }
    assert (
        client.post(
            "/draft/recommend", json={"opponentTeamId": "opp", "ourSide": "blue"}
        ).status_code
        == 200
    )
    schema = client.get("/openapi.json").json()
    assert "RosterResponse" in schema["components"]["schemas"]
