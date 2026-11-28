"""Train and evaluate the offline draft win-probability network."""

import argparse
import importlib
import json
import math
import random
from collections import defaultdict, deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

from app.config import DerivedSettings
from app.engine.draft_order import DRAFT_ORDER
from app.engine.scoring import Scorer
from app.engine.stats import blend_patches
from app.model.features import (
    DraftState,
    EncodedState,
    FeatureSpace,
    RosterHistory,
    player_history,
    team_elo_ratings,
)
from app.pipeline.aggregates import compute_aggregates
from app.pipeline.dataset import GameDraft

_STOPPING_SMOOTHING_WINDOW = 3


@dataclass(frozen=True)
class _PrefixExample:
    game: GameDraft
    state: DraftState
    label: float
    weight: float


class TrainingReport(BaseModel):
    """Held-out model and baseline metrics from a time-ordered split."""

    model_config = ConfigDict(frozen=True)

    train_games: int
    validation_games: int
    stopping_games: int
    best_epoch: int
    train_examples: int
    validation_examples: int
    seed: int
    device: str
    metrics: dict[str, dict[str, float | None]]


def _is_decided(game: GameDraft) -> bool:
    return game.finished and game.winner_team_id in {game.blue_team_id, game.red_team_id}


def _date_key(game: GameDraft) -> float:
    try:
        value = datetime.fromisoformat(game.started_at.replace("Z", "+00:00"))
        if value.tzinfo is None:
            value = value.replace(tzinfo=UTC)
        return value.timestamp()
    except ValueError:
        return float("-inf")


def _series_key(games: list[GameDraft]) -> tuple[float, str]:
    dated_games = [_date_key(game) for game in games]
    known_dates = [value for value in dated_games if value != float("-inf")]
    return min(known_dates, default=float("-inf")), games[0].series_id


def _ordered_series(games: list[GameDraft]) -> list[str]:
    series_games: dict[str, list[GameDraft]] = defaultdict(list)
    for game in games:
        series_games[game.series_id].append(game)
    return sorted(series_games, key=lambda series_id: _series_key(series_games[series_id]))


def time_split(
    games: list[GameDraft], train_fraction: float = 0.85
) -> tuple[list[GameDraft], list[GameDraft]]:
    """Split chronologically by series, reserving the newest series for validation."""
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be between 0 and 1")
    ordered_series = _ordered_series(games)
    if len(ordered_series) < 2:
        return sorted(games, key=lambda game: (_date_key(game), game.game_number, game.game_id)), []
    split_at = min(max(int(len(ordered_series) * train_fraction), 1), len(ordered_series) - 1)
    training_series = set(ordered_series[:split_at])
    ordered_games = sorted(
        games,
        key=lambda game: (_date_key(game), game.series_id, game.game_number, game.game_id),
    )
    return (
        [game for game in ordered_games if game.series_id in training_series],
        [game for game in ordered_games if game.series_id not in training_series],
    )


def random_split(
    games: list[GameDraft], train_fraction: float = 0.85, *, seed: int = 42
) -> tuple[list[GameDraft], list[GameDraft]]:
    """Split by series with random assignment, keeping each series intact.

    Unlike ``time_split`` this ignores chronology, so train and validation are
    drawn from the same patch/meta distribution. It exists to probe the data
    ceiling (how much of the low validation AUC is distribution shift versus
    genuinely weak signal), not for production training.
    """
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be between 0 and 1")
    ordered_series = _ordered_series(games)
    if len(ordered_series) < 2:
        return sorted(games, key=lambda game: (_date_key(game), game.game_number, game.game_id)), []
    shuffled = list(ordered_series)
    random.Random(seed).shuffle(shuffled)
    split_at = min(max(int(len(shuffled) * train_fraction), 1), len(shuffled) - 1)
    training_series = set(shuffled[:split_at])
    ordered_games = sorted(
        games,
        key=lambda game: (_date_key(game), game.series_id, game.game_number, game.game_id),
    )
    return (
        [game for game in ordered_games if game.series_id in training_series],
        [game for game in ordered_games if game.series_id not in training_series],
    )


_SplitFn = Callable[[list[GameDraft], float], tuple[list[GameDraft], list[GameDraft]]]


def _resolve_split(strategy: str, seed: int) -> _SplitFn:
    if strategy == "chronological":
        return time_split
    if strategy == "random":
        return lambda games, train_fraction: random_split(games, train_fraction, seed=seed)
    raise ValueError(f"Unsupported split strategy: {strategy!r} (use 'chronological' or 'random')")


def _prefix_examples(
    games: list[GameDraft],
    pre_game_elo: dict[str, tuple[float, float]] | None = None,
    roster: dict[str, RosterHistory] | None = None,
) -> list[_PrefixExample]:
    examples: list[_PrefixExample] = []
    pre_game_elo = pre_game_elo or {}
    roster = roster or {}
    for game in games:
        if not _is_decided(game):
            continue
        label = 1.0 if game.winner_team_id == game.blue_team_id else 0.0
        action_ids = [action.champion_id for action in game.actions[: len(DRAFT_ORDER)]]
        blue_elo, red_elo = pre_game_elo.get(game.game_id, (None, None))
        history = roster.get(game.game_id)
        roster_fields: dict[str, Any] = (
            {
                "blue_player_elo": history.blue_elo,
                "red_player_elo": history.red_elo,
                "blue_comfort": history.blue_comfort,
                "red_comfort": history.red_comfort,
            }
            if history
            else {}
        )
        for prefix_length in range(len(action_ids) + 1):
            examples.append(
                _PrefixExample(
                    game=game,
                    state=DraftState(
                        patch=game.patch or "unknown",
                        actions=action_ids[:prefix_length],
                        blue_team_id=game.blue_team_id,
                        red_team_id=game.red_team_id,
                        blue_elo=blue_elo,
                        red_elo=red_elo,
                        **roster_fields,
                    ),
                    label=label,
                    weight=prefix_length / len(DRAFT_ORDER),
                )
            )
    return examples


def _out_of_fold_features(
    examples: list[_PrefixExample],
    games: list[GameDraft],
    ddragon: dict[str, Any] | None = None,
) -> list[EncodedState]:
    """Encode each example with historical stats fitted outside its series fold."""
    series_games: dict[str, list[GameDraft]] = defaultdict(list)
    for game in games:
        series_games[game.series_id].append(game)
    ordered_series = sorted(
        series_games, key=lambda series_id: _series_key(series_games[series_id])
    )
    if not ordered_series:
        return []

    fold_count = min(5, len(ordered_series))
    series_folds = {series_id: index % fold_count for index, series_id in enumerate(ordered_series)}
    examples_by_fold: dict[int, list[tuple[int, _PrefixExample]]] = defaultdict(list)
    for index, example in enumerate(examples):
        fold = series_folds.get(example.game.series_id)
        if fold is None:
            raise ValueError(
                f"Example series {example.game.series_id!r} is absent from training games"
            )
        examples_by_fold[fold].append((index, example))

    encoded: list[EncodedState | None] = [None] * len(examples)
    for fold, fold_examples in examples_by_fold.items():
        history_games = [game for game in games if series_folds[game.series_id] != fold]
        feature_space = FeatureSpace.fit(
            history_games,
            ddragon,
            vocabulary_games=games,
        )
        for index, example in fold_examples:
            encoded[index] = feature_space.encode(example.state)

    if any(value is None for value in encoded):
        raise RuntimeError("Not all training examples received out-of-fold features")
    return [value for value in encoded if value is not None]


def _torch_module() -> Any:
    try:
        return importlib.import_module("torch")
    except ImportError as error:
        raise RuntimeError("Install a PyTorch extra, for example `uv sync --extra cpu`") from error


def _metrics(labels: list[float], probabilities: list[float]) -> dict[str, float | None]:
    if not labels:
        return {"accuracy": None, "auc": None, "log_loss": None}
    accuracy = sum(
        (probability >= 0.5) == bool(label)
        for label, probability in zip(labels, probabilities, strict=True)
    )
    positives = sum(label == 1.0 for label in labels)
    negatives = len(labels) - positives
    auc: float | None = None
    if positives and negatives:
        ordered = sorted(zip(probabilities, labels, strict=True))
        rank_sum = 0.0
        index = 0
        while index < len(ordered):
            end = index + 1
            while end < len(ordered) and ordered[end][0] == ordered[index][0]:
                end += 1
            average_rank = ((index + 1) + end) / 2.0
            rank_sum += average_rank * sum(label for _, label in ordered[index:end])
            index = end
        auc = (rank_sum - positives * (positives + 1) / 2) / (positives * negatives)
    log_loss = -sum(
        label * math.log(max(probability, 1e-7))
        + (1 - label) * math.log(max(1 - probability, 1e-7))
        for label, probability in zip(labels, probabilities, strict=True)
    ) / len(labels)
    return {"accuracy": accuracy / len(labels), "auc": auc, "log_loss": log_loss}


def _scorer_probabilities(
    examples: list[_PrefixExample], train_games: list[GameDraft]
) -> list[float]:
    aggregates = compute_aggregates(train_games)
    if not aggregates:
        return [0.5] * len(examples)
    champion_names = {
        action.champion_id: action.champion_name for game in train_games for action in game.actions
    }
    blended_by_patch: dict[str | None, Any] = {}
    champion_roles = FeatureSpace.fit(train_games).champion_roles
    scorers: dict[tuple[str, str, str], Scorer] = {}
    probabilities: list[float] = []
    for example in examples:
        game = example.game
        key = (game.patch, game.blue_team_id, game.red_team_id)
        scorer = scorers.get(key)
        if scorer is None:
            patch = game.patch if game.patch in aggregates else None
            stats = blended_by_patch.get(patch)
            if stats is None:
                stats = blend_patches(aggregates, patch=patch)
                blended_by_patch[patch] = stats
            scorer = Scorer(
                stats,
                champion_names=champion_names,
                our_team_id=game.blue_team_id,
                our_team_name=game.blue_team_id,
                opponent_team_id=game.red_team_id,
                opponent_team_name=game.red_team_id,
                our_side="blue",
                champion_roles=champion_roles,
            )
            scorers[key] = scorer
        blue_picks: list[str] = []
        red_picks: list[str] = []
        for index, champion_id in enumerate(example.state.actions):
            slot = DRAFT_ORDER[index]
            if slot.action == "pick":
                (blue_picks if slot.side == "blue" else red_picks).append(champion_id)
        probabilities.append(scorer.game_win_probability(blue_picks, red_picks))
    return probabilities


def _elo_probability(state: DraftState) -> float:
    blue, red = state.blue_elo or 1500.0, state.red_elo or 1500.0
    return 1.0 / (1.0 + 10.0 ** ((red - blue) / 400.0))


def _full_draft_lift(
    examples: list[_PrefixExample],
    model_probabilities: list[float],
    seed: int,
    resamples: int = 200,
) -> dict[str, float | None]:
    """Full-draft AUC of the model vs the Elo-only baseline, with a paired bootstrap CI on lift."""
    rows = [
        (example.label, probability, _elo_probability(example.state))
        for example, probability in zip(examples, model_probabilities, strict=True)
        if len(example.state.actions) == len(DRAFT_ORDER)
    ]

    def auc_pair(sample: list[tuple[float, float, float]]) -> tuple[float, float] | None:
        labels = [row[0] for row in sample]
        model_auc = _metrics(labels, [row[1] for row in sample])["auc"]
        elo_auc = _metrics(labels, [row[2] for row in sample])["auc"]
        return None if model_auc is None or elo_auc is None else (model_auc, elo_auc)

    base = auc_pair(rows)
    if base is None:
        return {
            "model_auc": None,
            "elo_auc": None,
            "lift": None,
            "lift_ci_low": None,
            "lift_ci_high": None,
        }
    rng = random.Random(seed)
    lifts = sorted(
        pair[0] - pair[1]
        for _ in range(resamples)
        if (pair := auc_pair(rng.choices(rows, k=len(rows)))) is not None
    )
    return {
        "model_auc": base[0],
        "elo_auc": base[1],
        "lift": base[0] - base[1],
        "lift_ci_low": lifts[int(0.025 * len(lifts))] if lifts else None,
        "lift_ci_high": lifts[int(0.975 * len(lifts)) - 1] if lifts else None,
    }


def _resolve_device(torch: Any, device: str) -> Any:
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available")
    if device not in {"cpu", "cuda"}:
        raise ValueError(f"Unsupported device: {device}")
    return torch.device(device)


def _stack_encoded(
    torch: Any, encoded_states: Sequence[EncodedState], device: Any
) -> tuple[Any, Any, Any, Any, Any]:
    """Stack a batch of EncodedState into the five tensors DraftNet.forward expects."""
    return (
        torch.tensor(
            [state.blue_picks for state in encoded_states], dtype=torch.long, device=device
        ),
        torch.tensor(
            [state.blue_bans for state in encoded_states], dtype=torch.long, device=device
        ),
        torch.tensor(
            [state.red_picks for state in encoded_states], dtype=torch.long, device=device
        ),
        torch.tensor([state.red_bans for state in encoded_states], dtype=torch.long, device=device),
        torch.tensor(
            [state.continuous for state in encoded_states], dtype=torch.float32, device=device
        ),
    )


def train_model(
    games: list[GameDraft],
    output_dir: Path,
    *,
    epochs: int = 100,
    hidden_size: int = 32,
    dropout: float = 0.5,
    embedding_dim: int = 16,
    learning_rate: float = 0.001,
    weight_decay: float = 1e-2,
    batch_size: int = 256,
    patience: int = 20,
    seed: int = 42,
    device: str = "auto",
    split: str = "chronological",
    ddragon: dict[str, Any] | None = None,
    verbose: bool = False,
) -> TrainingReport:
    """Train on decided games, validate on a held-out split, and save local artifacts.

    ``split`` selects how games are partitioned: ``"chronological"`` (default, production)
    reserves the newest series for validation; ``"random"`` assigns series at random, which
    removes patch/meta drift and is used only to probe the achievable data ceiling.
    All splits keep series together. Training examples use series-grouped out-of-fold history;
    stopping and validation features use a feature space fitted on the earlier fitting window.
    """
    torch = _torch_module()
    from app.model.net import DraftNet

    if verbose:
        print(f"Splitting {len(games)} games ({split} split)...", flush=True)
    split_fn = _resolve_split(split, seed)
    decided_games = [game for game in games if _is_decided(game)]
    train_games, validation_games = split_fn(decided_games, 0.85)
    if not train_games:
        raise ValueError("At least one decided game with draft actions is required")
    fit_games, stopping_games = split_fn(train_games, 0.85)
    if verbose:
        print(
            f"Split into {len(fit_games)} fit / {len(stopping_games)} stopping / "
            f"{len(validation_games)} validation games. Building prefix examples...",
            flush=True,
        )
    pre_game_elo, final_elo = team_elo_ratings(decided_games)
    roster = player_history(decided_games)
    train_examples = _prefix_examples(fit_games, pre_game_elo, roster)
    if not train_examples:
        raise ValueError("Training games do not contain draft actions")
    stopping_examples = _prefix_examples(stopping_games, pre_game_elo, roster)
    validation_examples = _prefix_examples(validation_games, pre_game_elo, roster)
    if verbose:
        print(
            f"Built {len(train_examples)} training examples. Fitting feature space on "
            f"{len(fit_games)} games...",
            flush=True,
        )
    # Final ratings include every decided game so inference starts from current team strength.
    feature_space = FeatureSpace.fit(fit_games, ddragon).model_copy(update={"team_elo": final_elo})
    if verbose:
        print(
            "Encoding training examples with 5-fold out-of-fold features "
            "(this is the slow step on large datasets)...",
            flush=True,
        )
    train_encoded = _out_of_fold_features(train_examples, fit_games, ddragon)
    train_labels = [example.label for example in train_examples]
    train_weights = [example.weight for example in train_examples]
    if verbose:
        print("Encoding stopping and validation features...", flush=True)
    stopping_encoded = [feature_space.encode(example.state) for example in stopping_examples]
    stopping_labels = [example.label for example in stopping_examples]
    stopping_weights = [example.weight for example in stopping_examples]
    validation_encoded = [feature_space.encode(example.state) for example in validation_examples]
    validation_labels = [example.label for example in validation_examples]

    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    target_device = _resolve_device(torch, device)
    if verbose:
        print(
            f"Training on device={target_device}: {len(fit_games)} fit games "
            f"({len(train_examples)} examples), {len(stopping_games)} stopping games, "
            f"{len(validation_games)} validation games, up to {epochs} epochs, "
            f"patience={patience}"
        )
    model = DraftNet(
        feature_space.num_champions,
        feature_space.continuous_size,
        embedding_dim=embedding_dim,
        hidden_size=hidden_size,
        dropout=dropout,
    ).to(target_device)
    blue_picks_tensor, blue_bans_tensor, red_picks_tensor, red_bans_tensor, continuous_tensor = (
        _stack_encoded(torch, train_encoded, target_device)
    )
    labels_tensor = torch.tensor(train_labels, dtype=torch.float32, device=target_device)
    weights_tensor = torch.tensor(train_weights, dtype=torch.float32, device=target_device)
    stopping_tensors = _stack_encoded(torch, stopping_encoded, target_device)
    stopping_labels_tensor = torch.tensor(
        stopping_labels, dtype=torch.float32, device=target_device
    )
    stopping_weights_tensor = torch.tensor(
        stopping_weights, dtype=torch.float32, device=target_device
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    best_loss = float("inf")
    best_epoch = epochs
    best_state: dict[str, Any] | None = None
    train_size = labels_tensor.shape[0]
    recent_stopping_losses: deque[float] = deque(maxlen=_STOPPING_SMOOTHING_WINDOW)
    for epoch in range(1, epochs + 1):
        model.train()
        permutation = torch.randperm(train_size, device=target_device)
        epoch_loss_total = 0.0
        for start in range(0, train_size, batch_size):
            batch_indices = permutation[start : start + batch_size]
            optimizer.zero_grad()
            probabilities = model(
                blue_picks_tensor[batch_indices],
                blue_bans_tensor[batch_indices],
                red_picks_tensor[batch_indices],
                red_bans_tensor[batch_indices],
                continuous_tensor[batch_indices],
            )
            batch_weights = weights_tensor[batch_indices]
            per_example_loss = torch.nn.functional.binary_cross_entropy(
                probabilities, labels_tensor[batch_indices], reduction="none"
            )
            loss = (per_example_loss * batch_weights).sum() / batch_weights.sum().clamp_min(1e-8)
            loss.backward()
            optimizer.step()
            epoch_loss_total += float(loss.detach()) * batch_indices.shape[0]
        train_loss = epoch_loss_total / train_size
        if not stopping_encoded:
            if verbose:
                print(f"Epoch {epoch}/{epochs}: train_loss={train_loss:.4f}")
            continue
        model.eval()
        with torch.inference_mode():
            stopping_probabilities = model(*stopping_tensors)
            stopping_per_example_loss = torch.nn.functional.binary_cross_entropy(
                stopping_probabilities, stopping_labels_tensor, reduction="none"
            )
            stopping_loss = float(
                (stopping_per_example_loss * stopping_weights_tensor).sum()
                / stopping_weights_tensor.sum().clamp_min(1e-8)
            )
        recent_stopping_losses.append(stopping_loss)
        smoothed_stopping_loss = sum(recent_stopping_losses) / len(recent_stopping_losses)
        is_best = smoothed_stopping_loss < best_loss
        if is_best:
            best_loss = smoothed_stopping_loss
            best_epoch = epoch
            best_state = {
                name: value.detach().clone() for name, value in model.state_dict().items()
            }
        if verbose:
            marker = " (best)" if is_best else ""
            print(
                f"Epoch {epoch}/{epochs}: train_loss={train_loss:.4f} "
                f"stopping_loss={stopping_loss:.4f}{marker}"
            )
        if not is_best and epoch - best_epoch >= patience:
            if verbose:
                print(f"Stopped early at epoch {epoch} (best epoch {best_epoch})")
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    if verbose:
        print(f"Training finished: best epoch {best_epoch}, best stopping_loss={best_loss:.4f}")

    model.eval()
    if validation_encoded:
        with torch.inference_mode():
            validation_tensors = _stack_encoded(torch, validation_encoded, target_device)
            model_probabilities = [
                float(value) for value in model(*validation_tensors).detach().cpu().tolist()
            ]
    else:
        model_probabilities = []

    blue_rate = sum(game.winner_team_id == game.blue_team_id for game in train_games) / len(
        train_games
    )
    baseline_probabilities = [blue_rate] * len(validation_labels)
    scorer_probabilities = _scorer_probabilities(validation_examples, train_games)
    report = TrainingReport(
        train_games=len(train_games),
        validation_games=len(validation_games),
        stopping_games=len(stopping_games),
        best_epoch=best_epoch,
        train_examples=len(train_examples),
        validation_examples=len(validation_examples),
        seed=seed,
        device=str(target_device),
        metrics={
            "model": _metrics(validation_labels, model_probabilities),
            "blue_rate_baseline": _metrics(validation_labels, baseline_probabilities),
            "statistical_scorer": _metrics(validation_labels, scorer_probabilities),
            "elo_baseline": _metrics(
                validation_labels, [_elo_probability(e.state) for e in validation_examples]
            ),
            "full_draft_vs_elo": _full_draft_lift(validation_examples, model_probabilities, seed),
        },
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.cpu().state_dict(),
            "num_champions": feature_space.num_champions,
            "continuous_size": feature_space.continuous_size,
            "embedding_dim": embedding_dim,
            "hidden_size": hidden_size,
            "dropout": dropout,
            "feature_space": feature_space.model_dump(mode="json"),
        },
        output_dir / "model.pt",
    )
    (output_dir / "model_metrics.json").write_text(
        json.dumps(report.model_dump(mode="json"), indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def _read_traits(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return payload if isinstance(payload, dict) else {}


def main(argv: list[str] | None = None) -> None:
    """Train the offline model from locally built drafts.json data."""
    parser = argparse.ArgumentParser(description="Train Draft Coach's draft win model.")
    parser.add_argument("--drafts", type=Path, default=None)
    parser.add_argument("--ddragon", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    arguments = parser.parse_args(argv)

    output_dir = arguments.out or DerivedSettings().derived_dir
    drafts_path = arguments.drafts or output_dir / "drafts.json"
    ddragon_path = arguments.ddragon or output_dir / "ddragon.json"
    payload = json.loads(drafts_path.read_text(encoding="utf-8"))
    games = [GameDraft.model_validate(item) for item in payload]
    report = train_model(
        games,
        output_dir,
        epochs=arguments.epochs,
        hidden_size=arguments.hidden_size,
        dropout=arguments.dropout,
        weight_decay=arguments.weight_decay,
        learning_rate=arguments.learning_rate,
        batch_size=arguments.batch_size,
        patience=arguments.patience,
        seed=arguments.seed,
        device=arguments.device,
        ddragon=_read_traits(ddragon_path),
        verbose=True,
    )
    print(json.dumps(report.model_dump(mode="json"), indent=2))


if __name__ == "__main__":
    main()
