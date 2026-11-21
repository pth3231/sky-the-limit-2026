import pytest
from fastapi.testclient import TestClient

from app.api.main import create_app
from app.engine.service import DraftEngine

BANS = ["c7", "c8", "c9", "c10", "c11", "c4"]


@pytest.fixture
def client(engine: DraftEngine) -> TestClient:
    return TestClient(create_app(engine))


def test_health(client: TestClient) -> None:
    assert client.get("/health").json() == {"status": "ok"}


def test_lists_use_camel_case(client: TestClient) -> None:
    teams = client.get("/teams").json()
    champions = client.get("/champions").json()

    assert teams[0] == {"teamId": "opp", "name": "Rivals"}
    assert {"championId": "c0", "name": "Champ 0", "role": None, "portraitUrl": None} in champions
    assert client.get("/patches").json() == ["16.1"]


def test_recommend_accepts_and_returns_camel_case(client: TestClient) -> None:
    response = client.post(
        "/draft/recommend",
        json={
            "opponentTeamId": "opp",
            "seriesFormat": "bo3",
            "ourSide": "blue",
            "previousGames": [],
            "actions": BANS,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["nextSlot"] == {"index": 6, "side": "blue", "action": "pick", "team": "us"}
    assert body["recommendation"]["championId"] == "c5"
    assert body["recommendation"]["contributingFactors"][0]["signal"] in {
        "strong",
        "some",
        "limited",
    }
    assert {"winRate", "gameWinRate", "opponentTendencies", "fearlessExcludedIds"} <= set(body)
    assert body["engine"] == "statistical"


@pytest.mark.parametrize(
    "payload",
    [
        {"opponentTeamId": "opp", "ourSide": "blue", "actions": ["c0", "c0"]},
        {"opponentTeamId": "nobody", "ourSide": "blue"},
        {"opponentTeamId": "opp", "ourSide": "blue", "patch": "1.0"},
    ],
)
def test_illegal_draft_state_returns_422(client: TestClient, payload: dict[str, object]) -> None:
    response = client.post("/draft/recommend", json=payload)

    assert response.status_code == 422
    details = response.json()["detail"]
    assert isinstance(details, list)
    assert {"loc", "msg", "type"} <= set(details[0])
