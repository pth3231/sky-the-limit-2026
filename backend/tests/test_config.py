from pathlib import Path

import pytest

from app.config import DEFAULT_DERIVED_DIR, Settings, resolve_data_dir


def test_environment_directory_overrides_candidates(tmp_path: Path) -> None:
    data_dir = tmp_path / "configured-data"
    (data_dir / "meta").mkdir(parents=True)

    resolved = resolve_data_dir(
        {"STL_DATA_DIR": str(data_dir)},
        candidates=[tmp_path / "fallback"],
    )

    assert resolved == data_dir


def test_uses_first_existing_candidate_with_meta(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    (second / "meta").mkdir(parents=True)
    (first / "meta").mkdir(parents=True)

    resolved = resolve_data_dir({}, candidates=[first, second])

    assert resolved == first


def test_missing_meta_directory_fails_clearly(tmp_path: Path) -> None:
    data_dir = tmp_path / "invalid-data"
    data_dir.mkdir()

    with pytest.raises(ValueError, match="meta"):
        resolve_data_dir({"STL_DATA_DIR": str(data_dir)}, candidates=[])


def test_settings_reads_stl_data_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    data_dir = tmp_path / "configured-data"
    (data_dir / "meta").mkdir(parents=True)
    monkeypatch.setenv("STL_DATA_DIR", str(data_dir))

    assert Settings().data_dir == data_dir


def test_settings_default_and_override_derived_dir_and_team(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_dir = tmp_path / "configured-data"
    (data_dir / "meta").mkdir(parents=True)
    monkeypatch.setenv("STL_DATA_DIR", str(data_dir))
    monkeypatch.delenv("DRAFT_COACH_DERIVED_DIR", raising=False)
    monkeypatch.delenv("CLOUD9_TEAM_ID", raising=False)

    defaults = Settings()
    assert defaults.derived_dir == DEFAULT_DERIVED_DIR
    assert defaults.cloud9_team_id is None

    monkeypatch.setenv("DRAFT_COACH_DERIVED_DIR", str(tmp_path / "derived"))
    monkeypatch.setenv("CLOUD9_TEAM_ID", "47351")
    configured = Settings()
    assert configured.derived_dir == tmp_path / "derived"
    assert configured.cloud9_team_id == "47351"
