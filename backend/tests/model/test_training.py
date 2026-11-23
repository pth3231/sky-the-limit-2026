from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest
import torch

from app.engine.draft_order import DRAFT_ORDER
from app.model.features import DraftState
from app.model.net import DraftNet
from app.model.predictor import WinPredictor
from app.model.train import (
    _out_of_fold_features,
    _prefix_examples,
    random_split,
    time_split,
    train_model,
)
from app.pipeline.dataset import GameDraft
from app.pipeline.drafts import DraftAction, PlayerLine


def _planted_game(make_game: Any, index: int, blue_has_winner: bool) -> GameDraft:
    actions = []
    for slot in DRAFT_ORDER:
        if slot.action == "ban":
            champion_id = f"{slot.side}-ban-{slot.index}"
        elif slot.side == "blue" and slot.index == 6:
            champion_id = "carry-c0" if blue_has_winner else "carry-c1"
        else:
            champion_id = f"{slot.side}-pick-{slot.index}"
        actions.append(
            DraftAction(
                sequence_number=slot.index + 1,
                action_type=slot.action,
                team_id="c9" if slot.side == "blue" else "opp",
                champion_id=champion_id,
                champion_name=champion_id,
            )
        )
    base = make_game(winner="blue" if blue_has_winner else "red")
    return base.model_copy(
        update={
            "actions": actions,
            "started_at": f"2026-10-{index + 1:02d}T12:00:00Z",
        }
    )


def _oe_player_row(gid, side, pos, champ, **stats):
    row = {
        "gameid": gid,
        "datacompleteness": "complete",
        "date": "2026-11-01 10:00:00",
        "patch": "16.20",
        "side": side,
        "position": pos,
        "teamname": f"{side} Team",
        "teamid": f"oe-{side.lower()}-{gid}",
        "champion": champ,
        "kills": "0",
        "assists": "0",
        "teamkills": "10",
        "damageshare": "0.2",
        "totaldamagetaken": "15000",
        "visionscore": "30",
        "earnedgoldshare": "0.2",
        "result": "1" if side == "Blue" else "0",
    }
    row.update({key: str(value) for key, value in stats.items()})
    return row


def _oe_team_row(gid, side, bans, result):
    row = {
        "gameid": gid,
        "datacompleteness": "complete",
        "date": "2026-11-01 10:00:00",
        "patch": "16.20",
        "side": side,
        "position": "team",
        "teamname": f"{side} Team",
        "teamid": f"oe-{side.lower()}-{gid}",
        "champion": "",
        "result": result,
        "kills": "10",
        "assists": "0",
        "teamkills": "10",
        "damageshare": "0",
        "totaldamagetaken": "0",
        "visionscore": "0",
        "earnedgoldshare": "0",
    }
    for index, ban in enumerate(bans, start=1):
        row[f"ban{index}"] = ban
    return row


def _oe_complete_game(gameid: str, *, blue_wins: bool) -> list[Mapping[str, str]]:
    positions = ("top", "jng", "mid", "bot", "sup")
    rows: list[Mapping[str, str]] = []
    for side in ("Blue", "Red"):
        for pos in positions:
            rows.append(_oe_player_row(gameid, side, pos, f"{gameid}-{side}-{pos}"))
    blue_result = "1" if blue_wins else "0"
    red_result = "0" if blue_wins else "1"
    rows.append(
        _oe_team_row(
            gameid,
            "Blue",
            [f"{gameid}-Bb{index}" for index in range(1, 6)],
            blue_result,
        )
    )
    rows.append(
        _oe_team_row(
            gameid,
            "Red",
            [f"{gameid}-Rb{index}" for index in range(1, 6)],
            red_result,
        )
    )
    return rows


def test_train_model_runs_on_mixed_grid_and_oe_synthetic_games(
    make_game: Any, tmp_path: Path
) -> None:
    from app.pipeline.sources.oracles_elixir import build_champion_id_map, build_game

    grid = [_planted_game(make_game, index, index % 2 == 0) for index in range(12)]
    oe = []
    for index in range(12):
        rows = _oe_complete_game(f"m{index}", blue_wins=index % 2 == 0)
        game = build_game(rows, build_champion_id_map({}))
        assert game is not None
        oe.append(game)

    report = train_model([*grid, *oe], tmp_path, epochs=2, hidden_size=8, seed=11, device="cpu")

    assert report.metrics["model"]["auc"] is not None


def test_draft_net_returns_bounded_batch_probabilities() -> None:
    model = DraftNet(
        num_champions=10,
        continuous_size=4,
        sequence_length=20,
        embedding_dim=4,
        hidden_size=8,
        dropout=0.1,
        attention_heads=2,
    )
    draft_sequence = torch.full((3, 20), 10, dtype=torch.long)  # sentinel == num_champions

    probabilities = model(draft_sequence, torch.zeros((3, 4)))

    assert probabilities.shape == (3,)
    assert torch.all(probabilities >= 0)
    assert torch.all(probabilities <= 1)


def test_time_split_reserves_the_latest_games_for_validation(make_game: Any) -> None:
    games = [
        make_game().model_copy(update={"started_at": f"2026-10-{day:02d}T12:00:00Z"})
        for day in (3, 1, 2, 4)
    ]

    train, validation = time_split(games)

    assert [game.started_at for game in train] == [
        "2026-10-01T12:00:00Z",
        "2026-10-02T12:00:00Z",
        "2026-10-03T12:00:00Z",
    ]
    assert [game.started_at for game in validation] == ["2026-10-04T12:00:00Z"]


def test_time_split_keeps_all_games_from_a_series_together(make_game: Any) -> None:
    games = [
        make_game().model_copy(
            update={
                "series_id": series_id,
                "game_number": game_number,
                "started_at": f"2026-10-{day:02d}T12:00:00Z",
            }
        )
        for series_id, game_number, day in (
            ("series-a", 1, 1),
            ("series-a", 2, 2),
            ("series-b", 1, 3),
            ("series-c", 1, 4),
        )
    ]

    train, validation = time_split(games, train_fraction=0.67)

    assert {game.series_id for game in train}.isdisjoint({game.series_id for game in validation})
    assert {game.series_id for game in train} == {"series-a", "series-b"}
    assert {game.series_id for game in validation} == {"series-c"}


def test_time_split_orders_unknown_timestamps_deterministically(make_game: Any) -> None:
    games = [
        make_game().model_copy(update={"series_id": "series-z", "started_at": ""}),
        make_game().model_copy(
            update={"series_id": "series-dated", "started_at": "2026-10-01T12:00:00Z"}
        ),
        make_game().model_copy(update={"series_id": "series-a", "started_at": "invalid"}),
    ]

    train, validation = time_split(games, train_fraction=0.67)
    reversed_train, reversed_validation = time_split(list(reversed(games)), train_fraction=0.67)

    assert [game.series_id for game in train] == ["series-a", "series-z"]
    assert [game.series_id for game in validation] == ["series-dated"]
    assert [game.series_id for game in reversed_train] == [game.series_id for game in train]
    assert [game.series_id for game in reversed_validation] == [
        game.series_id for game in validation
    ]


def test_random_split_keeps_each_series_intact(make_game: Any) -> None:
    games = [
        make_game().model_copy(update={"series_id": f"series-{index % 4}", "game_number": index})
        for index in range(12)
    ]

    train, validation = random_split(games, train_fraction=0.5, seed=7)

    train_series = {game.series_id for game in train}
    validation_series = {game.series_id for game in validation}
    assert train_series.isdisjoint(validation_series)
    assert train and validation


def test_random_split_is_deterministic_and_seed_sensitive(make_game: Any) -> None:
    games = [
        make_game().model_copy(update={"series_id": f"series-{index}", "game_number": 1})
        for index in range(20)
    ]

    def partition(split: tuple[list[Any], list[Any]]) -> tuple[frozenset[str], frozenset[str]]:
        return (
            frozenset(game.game_id for game in split[0]),
            frozenset(game.game_id for game in split[1]),
        )

    assert partition(random_split(games, seed=1)) == partition(random_split(games, seed=1))
    assert partition(random_split(games, seed=1)) != partition(random_split(games, seed=2))


def test_random_split_respects_train_fraction(make_game: Any) -> None:
    games = [
        make_game().model_copy(update={"series_id": f"series-{index}", "game_number": 1})
        for index in range(20)
    ]

    train, validation = random_split(games, train_fraction=0.8, seed=3)

    assert len({game.series_id for game in train}) == 16
    assert len({game.series_id for game in validation}) == 4


def test_train_model_accepts_random_split_strategy(make_game: Any, tmp_path: Path) -> None:
    games = [_planted_game(make_game, index, index % 2 == 0) for index in range(24)]

    report = train_model(
        games, tmp_path, epochs=3, hidden_size=8, seed=17, device="cpu", split="random"
    )

    assert report.metrics["model"]["auc"] is not None
    assert report.validation_games >= 1


def test_train_model_rejects_unknown_split_strategy(make_game: Any, tmp_path: Path) -> None:
    games = [_planted_game(make_game, index, index % 2 == 0) for index in range(24)]

    with pytest.raises(ValueError, match="split"):
        train_model(games, tmp_path, epochs=1, seed=17, device="cpu", split="banana")


def test_out_of_fold_features_exclude_outcomes_and_performance_from_their_series(
    make_game: Any,
) -> None:
    games = [
        _planted_game(make_game, index, index % 2 == 0).model_copy(
            update={
                "series_id": series_id,
                "started_at": f"2026-10-{index + 1:02d}T12:00:00Z",
                "lineups": [
                    PlayerLine(
                        team_id="c9",
                        role="mid",
                        champion_id="blue-pick-6",
                        damage_share=0.2,
                        damage_taken=20_000,
                        vision_score=30,
                        gold_share=0.25,
                        kill_participation=0.6,
                    )
                ],
            }
        )
        for index, series_id in enumerate(("series-a", "series-a", "series-b", "series-c"))
    ]
    example = _prefix_examples([games[0]])[-1]

    original = _out_of_fold_features([example], games)[0]
    perturbed = [
        game.model_copy(
            update={
                "winner_team_id": game.red_team_id,
                "lineups": [
                    game.lineups[0].model_copy(
                        update={
                            "damage_share": 0.9,
                            "damage_taken": 90_000,
                            "vision_score": 90,
                            "gold_share": 0.9,
                            "kill_participation": 0.9,
                        }
                    )
                ],
            }
        )
        if game.series_id == "series-a"
        else game
        for game in games
    ]
    changed = _out_of_fold_features([example], perturbed)[0]

    assert changed == original


def test_out_of_fold_features_are_deterministic_and_handle_one_series(make_game: Any) -> None:
    game = _planted_game(make_game, 0, True)
    examples = _prefix_examples([game])

    first = _out_of_fold_features(examples, [game])
    second = _out_of_fold_features(examples, [game])

    assert first == second
    assert len({len(encoded.continuous) for encoded in first}) == 1


def test_training_learns_planted_winning_champion_and_loads_batch_predictor(
    make_game: Any,
    tmp_path: Path,
) -> None:
    games = [_planted_game(make_game, index, index % 2 == 0) for index in range(24)]

    report = train_model(
        games,
        tmp_path,
        epochs=50,
        hidden_size=8,
        seed=17,
        device="cpu",
    )

    predictor = WinPredictor.load(tmp_path / "model.pt")
    winning = DraftState(patch="16.1", actions=[action.champion_id for action in games[0].actions])
    losing = DraftState(patch="16.1", actions=[action.champion_id for action in games[1].actions])
    probabilities = predictor.predict([winning, losing])

    assert probabilities[0] > probabilities[1]
    assert report.train_games == 20
    assert report.validation_games == 4
    assert report.stopping_games == 3
    assert 1 <= report.best_epoch <= 50
    assert report.metrics["model"]["auc"] is not None
    assert (tmp_path / "model_metrics.json").is_file()


def test_train_model_saves_embedding_checkpoint_fields(make_game: Any, tmp_path: Path) -> None:
    games = [_planted_game(make_game, index, index % 2 == 0) for index in range(24)]

    train_model(games, tmp_path, epochs=1, seed=17, device="cpu")

    checkpoint = torch.load(tmp_path / "model.pt", weights_only=True)
    assert checkpoint["num_champions"] > 0
    assert checkpoint["continuous_size"] > 0
    assert checkpoint["embedding_dim"] == 16
    assert checkpoint["sequence_length"] == len(DRAFT_ORDER)
    assert "input_size" not in checkpoint


def test_train_model_takes_multiple_optimizer_steps_per_epoch_with_small_batch_size(
    make_game: Any, tmp_path: Path, monkeypatch: Any
) -> None:
    games = [_planted_game(make_game, index, index % 2 == 0) for index in range(24)]
    step_calls = 0
    original_step = torch.optim.Adam.step

    def counting_step(self: Any, *args: Any, **kwargs: Any) -> Any:
        nonlocal step_calls
        step_calls += 1
        return original_step(self, *args, **kwargs)

    monkeypatch.setattr(torch.optim.Adam, "step", counting_step)

    train_model(
        games,
        tmp_path,
        epochs=1,
        hidden_size=8,
        seed=17,
        device="cpu",
        batch_size=50,
    )

    assert step_calls > 1


def test_train_model_defaults_use_tighter_regularization(make_game: Any, tmp_path: Path) -> None:
    games = [_planted_game(make_game, index, index % 2 == 0) for index in range(24)]

    train_model(games, tmp_path, epochs=1, seed=17, device="cpu")

    checkpoint = torch.load(tmp_path / "model.pt", weights_only=True)
    assert checkpoint["hidden_size"] == 32
    assert checkpoint["dropout"] == 0.5


def test_train_model_is_silent_by_default(make_game: Any, tmp_path: Path, capsys: Any) -> None:
    games = [_planted_game(make_game, index, index % 2 == 0) for index in range(24)]

    train_model(games, tmp_path, epochs=5, hidden_size=8, seed=17, device="cpu")

    assert capsys.readouterr().out == ""


def test_train_model_prints_progress_when_verbose(
    make_game: Any, tmp_path: Path, capsys: Any
) -> None:
    games = [_planted_game(make_game, index, index % 2 == 0) for index in range(24)]

    train_model(games, tmp_path, epochs=5, hidden_size=8, seed=17, device="cpu", verbose=True)

    output = capsys.readouterr().out
    assert "device" in output.lower()
    assert "epoch 1" in output.lower()
    assert "epoch 5" in output.lower()


def test_prefix_examples_weight_scales_with_locked_in_actions(make_game: Any) -> None:
    game = _planted_game(make_game, 0, True)

    examples = _prefix_examples([game])

    weights_by_prefix_length = {len(example.state.actions): example.weight for example in examples}
    assert weights_by_prefix_length[0] == 0.0
    assert weights_by_prefix_length[len(DRAFT_ORDER)] == 1.0
    assert weights_by_prefix_length[10] == 10 / len(DRAFT_ORDER)


def test_train_model_uses_weighted_loss_without_crashing(make_game: Any, tmp_path: Path) -> None:
    games = [_planted_game(make_game, index, index % 2 == 0) for index in range(24)]

    report = train_model(games, tmp_path, epochs=3, hidden_size=8, seed=17, device="cpu")

    assert report.metrics["model"]["auc"] is not None


def test_scorer_probabilities_caches_blend_patches_by_patch_not_by_team_pair(
    make_game: Any, monkeypatch: Any
) -> None:
    """Regression test: blend_patches depends only on `patch`, not on team identity.

    Calling it fresh for every (patch, team, team) combination wastes work proportional to
    the number of distinct team pairs, not the number of distinct patches — on datasets with
    many teams and a long patch history this made validation-metric computation dramatically
    slower and more memory-hungry than necessary.
    """
    from app.model import train as train_module

    games = []
    for index in range(6):
        blue_wins = index % 2 == 0
        blue_team_id, red_team_id = f"team-{index}", f"opp-{index}"
        game = _planted_game(make_game, index, blue_wins)
        actions = [
            action.model_copy(
                update={
                    "team_id": blue_team_id if action.team_id == "c9" else red_team_id,
                }
            )
            for action in game.actions
        ]
        games.append(
            game.model_copy(
                update={
                    "blue_team_id": blue_team_id,
                    "red_team_id": red_team_id,
                    "winner_team_id": blue_team_id if blue_wins else red_team_id,
                    "actions": actions,
                }
            )
        )

    all_examples = _prefix_examples(games)
    full_draft_length = len(DRAFT_ORDER)
    examples = [
        example for example in all_examples if len(example.state.actions) == full_draft_length
    ]
    assert len(examples) == len(games)

    call_count = 0
    original_blend_patches = train_module.blend_patches

    def counting_blend_patches(*args: Any, **kwargs: Any) -> Any:
        nonlocal call_count
        call_count += 1
        return original_blend_patches(*args, **kwargs)

    monkeypatch.setattr(train_module, "blend_patches", counting_blend_patches)

    train_module._scorer_probabilities(examples, games)

    distinct_patches = len({game.patch for game in games})
    assert call_count <= distinct_patches
