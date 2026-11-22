from pathlib import Path
from typing import Any

from app.model.train import TrainingReport


def _report(auc: float | None, best_epoch: int = 5) -> TrainingReport:
    return TrainingReport(
        train_games=10,
        validation_games=2,
        stopping_games=2,
        best_epoch=best_epoch,
        train_examples=100,
        validation_examples=20,
        seed=42,
        device="cpu",
        metrics={
            "model": {"accuracy": 0.5, "auc": auc, "log_loss": 0.69},
            "blue_rate_baseline": {"accuracy": 0.5, "auc": 0.5, "log_loss": 0.69},
            "statistical_scorer": {"accuracy": 0.5, "auc": 0.52, "log_loss": 0.69},
        },
    )


def test_run_tuning_evaluates_every_combination_and_sorts_by_auc(monkeypatch: Any) -> None:
    import app.model.tune as tune_module

    calls: list[dict[str, Any]] = []

    def fake_train_model(games: Any, output_dir: Path, **kwargs: Any) -> TrainingReport:
        calls.append(kwargs)
        auc_by_dropout = {0.2: 0.55, 0.3: 0.60, 0.4: None}
        return _report(auc_by_dropout[kwargs["dropout"]])

    monkeypatch.setattr(tune_module, "train_model", fake_train_model)

    trials = tune_module.run_tuning(
        [],
        Path("/tmp/does-not-matter"),
        dropout=[0.2, 0.3, 0.4],
        weight_decay=[1e-3],
        learning_rate=[0.001],
        hidden_size=[128],
        embedding_dim=[16],
    )

    assert len(calls) == 3
    assert {call["dropout"] for call in calls} == {0.2, 0.3, 0.4}
    assert [trial.dropout for trial in trials] == [0.3, 0.2, 0.4]
    assert trials[0].model_auc == 0.60
    assert trials[-1].model_auc is None
