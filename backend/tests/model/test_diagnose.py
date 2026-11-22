from typing import Any

from app.model.diagnose import patch_drift


def test_patch_drift_flags_fully_unseen_validation_patches(make_game: Any) -> None:
    train = [make_game(patch="16.1") for _ in range(8)]
    validation = [make_game(patch="16.5") for _ in range(2)]

    report = patch_drift(train, validation)

    assert report.train_counts == {"16.1": 8}
    assert report.validation_counts == {"16.5": 2}
    assert report.validation_unseen_fraction == 1.0
    assert report.jensen_shannon_divergence > 0.9


def test_patch_drift_zero_for_identical_distributions(make_game: Any) -> None:
    train = [make_game(patch="16.1") for _ in range(5)]
    validation = [make_game(patch="16.1") for _ in range(5)]

    report = patch_drift(train, validation)

    assert report.validation_unseen_fraction == 0.0
    assert report.jensen_shannon_divergence == 0.0


def test_patch_drift_partial_overlap_is_between_the_extremes(make_game: Any) -> None:
    train = [make_game(patch="16.1") for _ in range(6)]
    validation = [
        *(make_game(patch="16.1") for _ in range(2)),
        *(make_game(patch="16.5") for _ in range(2)),
    ]

    report = patch_drift(train, validation)

    assert report.validation_unseen_fraction == 0.5
    assert 0.0 < report.jensen_shannon_divergence < 1.0
