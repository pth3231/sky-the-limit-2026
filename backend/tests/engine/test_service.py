import csv
import json
from pathlib import Path
from typing import Any

import pytest

from app.engine.errors import DraftStateError
from app.engine.models import DraftRequest
from app.engine.service import DraftEngine, load_engine, resolve_our_team_id
from app.pipeline.aggregates import compute_aggregates
from app.pipeline.dataset import GameDraft
from app.pipeline.meta import Champion, Team

BANS = ["c7", "c8", "c9", "c10", "c11", "c4"]


def _request(**overrides: Any) -> DraftRequest:
    payload: dict[str, Any] = {"opponentTeamId": "opp", "ourSide": "blue", "actions": []}
    payload.update(overrides)
    return DraftRequest.model_validate(payload)


def test_our_slot_gets_a_legal_recommendation_and_alternatives(engine: DraftEngine) -> None:
    response = engine.recommend(_request(actions=BANS))

    assert response.next_slot is not None
    assert (response.next_slot.index, response.next_slot.team) == (6, "us")
    assert response.recommendation is not None
    assert len(response.alternatives) == 4
    suggested = [response.recommendation, *response.alternatives]
    assert not {candidate.champion_id for candidate in suggested} & set(BANS)
    assert [candidate.score for candidate in suggested] == sorted(
        (candidate.score for candidate in suggested), reverse=True
    )
    assert response.recommendation.champion_id == "c5"
    assert 0 < response.win_rate < 1
    assert response.contributing_factors == response.recommendation.contributing_factors


def test_fearless_excluded_champions_are_never_suggested(engine: DraftEngine) -> None:
    previous_picks = {"c0", "c1", "c2", "c3", "c5", "c6", "c12", "c13", "c14", "c15"}
    previous = {
        "ourPicks": ["c5", "c0", "c1", "c2", "c3"],
        "opponentPicks": ["c6", "c12", "c13", "c14", "c15"],
        "result": "loss",
    }

    response = engine.recommend(_request(previousGames=[previous], actions=BANS))

    assert response.game_number == 2
    assert set(response.fearless_excluded_ids) == previous_picks
    assert response.recommendation is not None
    suggested = {response.recommendation.champion_id} | {
        candidate.champion_id for candidate in response.alternatives
    }
    assert not suggested & (previous_picks | set(BANS))
    assert set(response.unavailable_ids) == previous_picks | set(BANS)


def test_opponent_slot_shows_tendencies_without_recommendation(engine: DraftEngine) -> None:
    response = engine.recommend(_request(ourSide="red"))

    assert response.next_slot is not None
    assert response.next_slot.team == "opponent"
    assert response.recommendation is None
    assert response.alternatives == []
    assert response.opponent_tendencies[0].champion_id == "c4"
    assert response.opponent_tendencies[0].action == "ban"
    assert response.opponent_tendencies[0].label == "Rivals banned it in 20 of 20 games"


def test_ban_slot_recommends_a_ban(engine: DraftEngine) -> None:
    response = engine.recommend(_request())

    assert response.next_slot is not None
    assert response.next_slot.action == "ban"
    assert response.recommendation is not None
    assert response.recommendation.reason.startswith(("Takes away", "Low-impact ban"))


def test_complete_draft_still_reports_win_rates(engine: DraftEngine) -> None:
    response = engine.recommend(
        _request(actions=[f"c{number}" for number in range(20)], seriesFormat="bo1")
    )

    assert response.next_slot is None
    assert response.recommendation is None
    assert response.alternatives == []
    assert response.win_rate == response.game_win_rate
    assert 0 < response.game_win_rate < 1
    assert response.contributing_factors[0].label


def test_unknown_opponent_is_rejected(engine: DraftEngine) -> None:
    with pytest.raises(DraftStateError, match="Unknown opponent"):
        engine.recommend(_request(opponentTeamId="nobody"))


def test_our_team_cannot_be_its_own_opponent(engine: DraftEngine) -> None:
    with pytest.raises(DraftStateError, match="cannot play itself"):
        engine.recommend(_request(opponentTeamId="c9"))


def test_opponent_without_history_gets_a_plain_note(engine: DraftEngine) -> None:
    response = engine.recommend(_request(opponentTeamId="ghost", ourSide="red"))

    assert response.opponent_tendencies == []
    assert response.sample_note is not None
    assert "No recorded games for Ghost Team" in response.sample_note


def test_thin_opponent_history_gets_a_caution_note(make_game: Any) -> None:
    games = [
        *(make_game(blue="t1", red="t2") for _ in range(80)),
        make_game(blue="c9", red="opp", red_bans=["c4"]),
    ]
    champions = {
        f"c{number}": Champion(character_id=f"c{number}", name=f"Champ {number}", games_played=0)
        for number in range(24)
    }
    teams = {
        team_id: Team(team_id=team_id, name=name)
        for team_id, name in {
            "c9": "Cloud9",
            "opp": "Rivals",
            "t1": "Team One",
            "t2": "Team Two",
        }.items()
    }
    engine = DraftEngine(compute_aggregates(games), champions, teams, our_team_id="c9")

    response = engine.recommend(_request(ourSide="red"))

    assert response.sample_note is not None
    assert "only 1 recorded game for Rivals" in response.sample_note


def test_opponent_history_from_older_patch_is_available(make_game: Any) -> None:
    games = [
        *(make_game("16.2", blue="t1", red="t2") for _ in range(60)),
        *(make_game("16.1", blue="c9", red="opp", red_bans=["c4"]) for _ in range(10)),
    ]
    champions = {
        f"c{number}": Champion(character_id=f"c{number}", name=f"Champ {number}", games_played=0)
        for number in range(24)
    }
    teams = {
        team_id: Team(team_id=team_id, name=name)
        for team_id, name in {
            "c9": "Cloud9",
            "opp": "Rivals",
            "t1": "Team One",
            "t2": "Team Two",
        }.items()
    }
    engine = DraftEngine(compute_aggregates(games), champions, teams, our_team_id="c9")

    response = engine.recommend(_request(ourSide="red"))

    assert response.patches_used == ["16.2", "16.1"]
    assert response.opponent_tendencies[0].label == "Rivals banned it in 10 of 10 games"


def test_lists_opponents_with_games_champions_and_patches(engine: DraftEngine) -> None:
    assert [team.name for team in engine.opponents()] == ["Rivals", "Team One", "Team Two"]
    assert len(engine.champion_list()) == 24
    assert engine.patches() == ["16.1"]


def test_resolves_cloud9_by_name_unless_configured() -> None:
    teams = {"1": Team(team_id="1", name="Rivals"), "2": Team(team_id="2", name="Cloud9 Kia")}

    assert resolve_our_team_id(teams, None) == "2"
    assert resolve_our_team_id(teams, "1") == "1"
    with pytest.raises(ValueError, match="Cloud9"):
        resolve_our_team_id({"1": teams["1"]}, None)


def test_load_engine_reads_derived_aggregates_and_meta(
    tmp_path: Path, synthetic_drafts: list[GameDraft]
) -> None:
    derived = tmp_path / "derived"
    derived.mkdir()
    aggregates = compute_aggregates(synthetic_drafts)
    (derived / "aggregates.json").write_text(
        json.dumps({patch: item.model_dump(mode="json") for patch, item in aggregates.items()}),
        encoding="utf-8",
    )
    meta = tmp_path / "data" / "meta"
    meta.mkdir(parents=True)
    _write_csv(
        meta / "characters.csv",
        [{"title": "lol", "character_id": "c0", "name": "Champ 0", "games_played": "1"}],
    )
    _write_csv(
        meta / "teams.csv",
        [
            {"title": "lol", "team_id": "c9", "name": "Cloud9 Kia"},
            {"title": "lol", "team_id": "opp", "name": "Rivals"},
        ],
    )

    engine = load_engine(derived, tmp_path / "data", None, min_games=1)

    assert engine.patches() == ["16.1"]
    assert {champion.champion_id for champion in engine.champion_list()} >= {"c0", "c5"}
    assert engine.recommend(_request()).recommendation is not None


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
