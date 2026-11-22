"""Diagnose whether the chronological validation split caps achievable AUC.

Two signals, printed side by side:

1. Patch drift between the chronological train and validation windows. If
   validation games sit on patches barely present in training, meta shift
   caps how well any model can transfer.
2. Validation AUC under the chronological split versus a random (same-
   distribution) split, with identical model and hyperparameters. If the
   random split scores much higher, distribution shift — not model quality —
   is the ceiling.

Run on a machine with the built GRID data and a PyTorch extra installed:

    uv run python -m app.model.diagnose
"""

import argparse
import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.config import DerivedSettings
from app.model.train import time_split, train_model
from app.pipeline.dataset import GameDraft


@dataclass(frozen=True)
class PatchDriftReport:
    """How differently patches are distributed across train and validation."""

    train_counts: dict[str, int]
    validation_counts: dict[str, int]
    validation_unseen_fraction: float
    jensen_shannon_divergence: float


def _normalized(counts: dict[str, int], keys: list[str]) -> list[float]:
    total = sum(counts.values())
    if total == 0:
        return [0.0 for _ in keys]
    return [counts.get(key, 0) / total for key in keys]


def _jensen_shannon_divergence(train_counts: dict[str, int], val_counts: dict[str, int]) -> float:
    """Jensen-Shannon divergence in bits: 0.0 identical, 1.0 fully disjoint."""
    keys = sorted(set(train_counts) | set(val_counts))
    if not keys:
        return 0.0
    p = _normalized(train_counts, keys)
    q = _normalized(val_counts, keys)

    def kl(a: list[float], b: list[float]) -> float:
        return sum(
            a_i * math.log2(a_i / b_i) for a_i, b_i in zip(a, b, strict=True) if a_i > 0 and b_i > 0
        )

    m = [(p_i + q_i) / 2 for p_i, q_i in zip(p, q, strict=True)]
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def patch_drift(
    train_games: list[GameDraft], validation_games: list[GameDraft]
) -> PatchDriftReport:
    """Compare patch distributions of the train and validation game sets."""
    train_counts = dict(Counter(game.patch or "unknown" for game in train_games))
    validation_counts = dict(Counter(game.patch or "unknown" for game in validation_games))
    train_patches = set(train_counts)
    unseen = sum(1 for game in validation_games if (game.patch or "unknown") not in train_patches)
    unseen_fraction = unseen / len(validation_games) if validation_games else 0.0
    return PatchDriftReport(
        train_counts=train_counts,
        validation_counts=validation_counts,
        validation_unseen_fraction=unseen_fraction,
        jensen_shannon_divergence=_jensen_shannon_divergence(train_counts, validation_counts),
    )


@dataclass(frozen=True)
class SplitResult:
    """Validation metrics from one training run under a given split strategy."""

    strategy: str
    seed: int
    model_auc: float | None
    model_accuracy: float | None
    model_log_loss: float | None
    scorer_auc: float | None


def _run_split(
    games: list[GameDraft],
    output_dir: Path,
    *,
    strategy: str,
    seed: int,
    ddragon: dict[str, Any] | None,
    device: str,
    epochs: int,
) -> SplitResult:
    report = train_model(
        games,
        output_dir,
        epochs=epochs,
        seed=seed,
        device=device,
        split=strategy,
        ddragon=ddragon,
        verbose=False,
    )
    model = report.metrics["model"]
    return SplitResult(
        strategy=strategy,
        seed=seed,
        model_auc=model["auc"],
        model_accuracy=model["accuracy"],
        model_log_loss=model["log_loss"],
        scorer_auc=report.metrics["statistical_scorer"]["auc"],
    )


def _mean(values: list[float]) -> float | None:
    present = [value for value in values if value is not None]
    return sum(present) / len(present) if present else None


def run_diagnostic(
    games: list[GameDraft],
    output_dir: Path,
    *,
    seeds: list[int],
    ddragon: dict[str, Any] | None = None,
    device: str = "auto",
    epochs: int = 100,
) -> None:
    """Print the patch-drift report and the chronological-vs-random AUC comparison."""
    decided = [game for game in games if game.finished and game.winner_team_id]
    train_games, validation_games = time_split(decided)
    drift = patch_drift(train_games, validation_games)

    print("=== DATA CEILING DIAGNOSTIC ===")
    print(f"{len(decided)} decided games across {len({g.series_id for g in decided})} series\n")

    print("--- Patch drift (chronological split) ---")
    all_patches = sorted(set(drift.train_counts) | set(drift.validation_counts))
    train_total = sum(drift.train_counts.values()) or 1
    val_total = sum(drift.validation_counts.values()) or 1
    print(f"{'patch':<10}{'train%':>9}{'val%':>9}")
    for patch in all_patches:
        train_pct = 100 * drift.train_counts.get(patch, 0) / train_total
        val_pct = 100 * drift.validation_counts.get(patch, 0) / val_total
        flag = "  <- unseen in train" if patch not in drift.train_counts else ""
        print(f"{patch:<10}{train_pct:>8.1f}%{val_pct:>8.1f}%{flag}")
    print(
        f"\nvalidation games on patches absent from training: "
        f"{100 * drift.validation_unseen_fraction:.1f}%"
    )
    print(
        f"patch Jensen-Shannon divergence (0=identical, 1=disjoint): "
        f"{drift.jensen_shannon_divergence:.3f}\n"
    )

    print(f"--- Validation AUC by split ({len(seeds)} random seed(s)) ---")
    chronological = _run_split(
        games,
        output_dir,
        strategy="chronological",
        seed=seeds[0],
        ddragon=ddragon,
        device=device,
        epochs=epochs,
    )
    random_results = [
        _run_split(
            games,
            output_dir,
            strategy="random",
            seed=seed,
            ddragon=ddragon,
            device=device,
            epochs=epochs,
        )
        for seed in seeds
    ]
    random_auc = _mean([result.model_auc for result in random_results if result.model_auc])
    print(f"{'split':<16}{'model AUC':>11}{'scorer AUC':>12}")
    print(
        f"{'chronological':<16}{_fmt(chronological.model_auc):>11}"
        f"{_fmt(chronological.scorer_auc):>12}"
    )
    print(f"{'random (mean)':<16}{_fmt(random_auc):>11}{'':>12}")
    for result in random_results:
        print(f"  seed={result.seed:<10}{_fmt(result.model_auc):>9}")

    print("\n--- Interpretation ---")
    if random_auc is not None and chronological.model_auc is not None:
        delta = random_auc - chronological.model_auc
        print(f"random - chronological AUC delta: {delta:+.3f}")
        if delta > 0.03:
            print(
                "Random split scores materially higher: the chronological split (patch/meta "
                "drift) is capping achievable AUC. Fix the split/weighting strategy, not the model."
            )
        else:
            print(
                "Random and chronological are comparable: the signal itself is weak, not the "
                "split. More model tuning will not help; new data sources are the lever."
            )

    # Retrain production artifacts with the real (chronological) split so the diagnostic
    # does not leave random-split weights in place.
    _run_split(
        games,
        output_dir,
        strategy="chronological",
        seed=seeds[0],
        ddragon=ddragon,
        device=device,
        epochs=epochs,
    )


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return payload if isinstance(payload, dict) else {}


def main(argv: list[str] | None = None) -> None:
    """Run the data-ceiling diagnostic from locally built drafts.json data."""
    parser = argparse.ArgumentParser(description="Diagnose the draft model's data ceiling.")
    parser.add_argument("--drafts", type=Path, default=None)
    parser.add_argument("--ddragon", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 1, 7])
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    arguments = parser.parse_args(argv)

    output_dir = arguments.out or DerivedSettings().derived_dir
    drafts_path = arguments.drafts or output_dir / "drafts.json"
    ddragon_path = arguments.ddragon or output_dir / "ddragon.json"
    payload = json.loads(drafts_path.read_text(encoding="utf-8"))
    games = [GameDraft.model_validate(item) for item in payload]
    run_diagnostic(
        games,
        output_dir,
        seeds=arguments.seeds,
        ddragon=_read_json(ddragon_path),
        device=arguments.device,
        epochs=arguments.epochs,
    )


if __name__ == "__main__":
    main()
