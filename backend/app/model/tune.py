"""Grid-search sweep over DraftNet regularization and capacity hyperparameters.

Trains one model per combination of the given hyperparameter lists on the production
(chronological) split and reports validation AUC, so a workable regularization setting can be
found without manually bisecting one knob at a time across separate runs.

Run on a machine with the built GRID data and a PyTorch extra installed:

    uv run python -m app.model.tune
"""

import argparse
import itertools
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.config import DerivedSettings
from app.model.train import train_model
from app.pipeline.dataset import GameDraft


@dataclass(frozen=True)
class TuneTrial:
    """One hyperparameter combination's held-out validation results."""

    dropout: float
    weight_decay: float
    learning_rate: float
    hidden_size: int
    embedding_dim: int
    best_epoch: int
    model_auc: float | None
    model_accuracy: float | None
    model_log_loss: float | None
    scorer_auc: float | None


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def run_tuning(
    games: list[GameDraft],
    output_dir: Path,
    *,
    dropout: list[float],
    weight_decay: list[float],
    learning_rate: list[float],
    hidden_size: list[int],
    embedding_dim: list[int],
    epochs: int = 100,
    patience: int = 20,
    batch_size: int = 256,
    seed: int = 42,
    device: str = "auto",
    split: str = "chronological",
    ddragon: dict[str, Any] | None = None,
    verbose: bool = False,
) -> list[TuneTrial]:
    """Train one model per combination of the given hyperparameter lists, best AUC first."""
    combinations = list(
        itertools.product(dropout, weight_decay, learning_rate, hidden_size, embedding_dim)
    )
    trials: list[TuneTrial] = []
    for index, (
        trial_dropout,
        trial_weight_decay,
        trial_lr,
        trial_hidden,
        trial_embedding,
    ) in enumerate(combinations, start=1):
        print(
            f"\n>>> Trial {index}/{len(combinations)}: dropout={trial_dropout} "
            f"weight_decay={trial_weight_decay} learning_rate={trial_lr} "
            f"hidden_size={trial_hidden} embedding_dim={trial_embedding}",
            flush=True,
        )
        report = train_model(
            games,
            output_dir,
            epochs=epochs,
            seed=seed,
            device=device,
            split=split,
            ddragon=ddragon,
            hidden_size=trial_hidden,
            embedding_dim=trial_embedding,
            dropout=trial_dropout,
            weight_decay=trial_weight_decay,
            learning_rate=trial_lr,
            batch_size=batch_size,
            patience=patience,
            verbose=verbose,
        )
        model_metrics = report.metrics["model"]
        trial = TuneTrial(
            dropout=trial_dropout,
            weight_decay=trial_weight_decay,
            learning_rate=trial_lr,
            hidden_size=trial_hidden,
            embedding_dim=trial_embedding,
            best_epoch=report.best_epoch,
            model_auc=model_metrics["auc"],
            model_accuracy=model_metrics["accuracy"],
            model_log_loss=model_metrics["log_loss"],
            scorer_auc=report.metrics["statistical_scorer"]["auc"],
        )
        trials.append(trial)
        print(
            f"    -> best_epoch={trial.best_epoch} model_auc={_fmt(trial.model_auc)} "
            f"model_log_loss={_fmt(trial.model_log_loss)}",
            flush=True,
        )
    return sorted(trials, key=lambda trial: (trial.model_auc is None, -(trial.model_auc or 0.0)))


def _print_leaderboard(trials: list[TuneTrial]) -> None:
    print("\n--- Leaderboard (best model AUC first) ---")
    header = (
        f"{'dropout':>8}{'weight_decay':>13}{'lr':>9}{'hidden':>7}{'embed':>7}"
        f"{'epoch':>7}{'model_auc':>11}{'scorer_auc':>12}"
    )
    print(header)
    for trial in trials:
        print(
            f"{trial.dropout:>8}{trial.weight_decay:>13}{trial.learning_rate:>9}"
            f"{trial.hidden_size:>7}{trial.embedding_dim:>7}{trial.best_epoch:>7}"
            f"{_fmt(trial.model_auc):>11}{_fmt(trial.scorer_auc):>12}"
        )


def _read_traits(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return payload if isinstance(payload, dict) else {}


def main(argv: list[str] | None = None) -> None:
    """Run a hyperparameter sweep from locally built drafts.json data."""
    parser = argparse.ArgumentParser(
        description="Sweep DraftNet hyperparameters to find a workable regularization setting."
    )
    parser.add_argument("--drafts", type=Path, default=None)
    parser.add_argument("--ddragon", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--split", choices=("chronological", "random"), default="chronological")
    parser.add_argument("--dropout", type=float, nargs="+", default=[0.2, 0.25, 0.3, 0.35])
    parser.add_argument("--weight-decay", type=float, nargs="+", default=[1e-4, 3e-4, 1e-3])
    parser.add_argument("--learning-rate", type=float, nargs="+", default=[0.001])
    parser.add_argument("--hidden-size", type=int, nargs="+", default=[128])
    parser.add_argument("--embedding-dim", type=int, nargs="+", default=[16])
    arguments = parser.parse_args(argv)

    output_dir = arguments.out or DerivedSettings().derived_dir
    drafts_path = arguments.drafts or output_dir / "drafts.json"
    ddragon_path = arguments.ddragon or output_dir / "ddragon.json"
    payload = json.loads(drafts_path.read_text(encoding="utf-8"))
    games = [GameDraft.model_validate(item) for item in payload]

    trial_count = (
        len(arguments.dropout)
        * len(arguments.weight_decay)
        * len(arguments.learning_rate)
        * len(arguments.hidden_size)
        * len(arguments.embedding_dim)
    )
    print(f"Running {trial_count} trials on {len(games)} games...", flush=True)

    trials = run_tuning(
        games,
        output_dir,
        dropout=arguments.dropout,
        weight_decay=arguments.weight_decay,
        learning_rate=arguments.learning_rate,
        hidden_size=arguments.hidden_size,
        embedding_dim=arguments.embedding_dim,
        epochs=arguments.epochs,
        patience=arguments.patience,
        batch_size=arguments.batch_size,
        seed=arguments.seed,
        device=arguments.device,
        split=arguments.split,
        ddragon=_read_traits(ddragon_path),
        verbose=True,
    )
    _print_leaderboard(trials)

    best = trials[0]
    print(
        f"\n--- Retraining with the best trial's hyperparameters "
        f"(dropout={best.dropout}, weight_decay={best.weight_decay}, "
        f"learning_rate={best.learning_rate}, hidden_size={best.hidden_size}, "
        f"embedding_dim={best.embedding_dim}) to leave production artifacts in that state ---"
    )
    train_model(
        games,
        output_dir,
        epochs=arguments.epochs,
        seed=arguments.seed,
        device=arguments.device,
        split=arguments.split,
        ddragon=_read_traits(ddragon_path),
        hidden_size=best.hidden_size,
        embedding_dim=best.embedding_dim,
        dropout=best.dropout,
        weight_decay=best.weight_decay,
        learning_rate=best.learning_rate,
        batch_size=arguments.batch_size,
        patience=arguments.patience,
        verbose=True,
    )


if __name__ == "__main__":
    main()
