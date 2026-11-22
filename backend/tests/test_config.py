from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import pytest

import app.config as config_module
from app.config import DEFAULT_DERIVED_DIR, DerivedSettings, Settings, resolve_data_dir


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


def test_settings_default_to_scorer_primary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_dir = tmp_path / "configured-data"
    (data_dir / "meta").mkdir(parents=True)
    monkeypatch.setenv("STL_DATA_DIR", str(data_dir))
    monkeypatch.delenv("DRAFT_COACH_USE_MODEL", raising=False)

    assert Settings().use_model is False


def test_settings_enable_model_via_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    data_dir = tmp_path / "configured-data"
    (data_dir / "meta").mkdir(parents=True)
    monkeypatch.setenv("STL_DATA_DIR", str(data_dir))
    monkeypatch.setenv("DRAFT_COACH_USE_MODEL", "true")

    assert Settings().use_model is True


def test_settings_uses_backend_dotenv_file() -> None:
    assert Settings.model_config.get("env_file") == config_module.BACKEND_ENV_FILE


def test_settings_loads_dotenv_with_environment_precedence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dotenv_data = tmp_path / "dotenv-data"
    shell_data = tmp_path / "shell-data"
    (dotenv_data / "meta").mkdir(parents=True)
    (shell_data / "meta").mkdir(parents=True)
    env_file = tmp_path / ".env"
    env_file.write_text(
        f"STL_DATA_DIR={dotenv_data}\nDRAFT_COACH_DERIVED_DIR={tmp_path / 'dotenv-derived'}\n"
        "CLOUD9_TEAM_ID=dotenv-team\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("STL_DATA_DIR", str(shell_data))
    monkeypatch.delenv("DRAFT_COACH_DERIVED_DIR", raising=False)
    monkeypatch.delenv("CLOUD9_TEAM_ID", raising=False)

    settings_type = cast(Any, Settings)
    settings = settings_type(_env_file=env_file)

    assert settings.data_dir == shell_data
    assert settings.derived_dir == tmp_path / "dotenv-derived"
    assert settings.cloud9_team_id == "dotenv-team"


def test_derived_settings_loads_dotenv_without_grid_data(tmp_path: Path) -> None:
    output_dir = tmp_path / "derived"
    env_file = tmp_path / ".env"
    env_file.write_text(f"DRAFT_COACH_DERIVED_DIR={output_dir}\n", encoding="utf-8")

    derived_settings_type = cast(Any, DerivedSettings)
    assert derived_settings_type(_env_file=env_file).derived_dir == output_dir


def test_derived_settings_ignores_stl_data_dir_in_shared_dotenv(tmp_path: Path) -> None:
    output_dir = tmp_path / "derived"
    env_file = tmp_path / ".env"
    env_file.write_text(
        f"STL_DATA_DIR=../stl-hackathon-2026-grid-starter\nDRAFT_COACH_DERIVED_DIR={output_dir}\n",
        encoding="utf-8",
    )

    derived_settings_type = cast(Any, DerivedSettings)
    assert derived_settings_type(_env_file=env_file).derived_dir == output_dir


def test_default_data_lookup_uses_only_the_sibling_dataset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[Path] = []

    def capture_candidates(env: Mapping[str, str], candidates: Sequence[Path]) -> Path:
        captured.extend(candidates)
        return candidates[0]

    monkeypatch.setattr(config_module, "resolve_data_dir", capture_candidates)

    resolved = config_module._default_data_dir()
    project_root = Path(config_module.__file__).resolve().parents[2]

    assert resolved == project_root.parent / "stl-hackathon-2026-grid-starter"
    assert captured == [resolved]


def test_settings_default_and_override_derived_dir_and_team(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_dir = tmp_path / "configured-data"
    (data_dir / "meta").mkdir(parents=True)
    monkeypatch.setenv("STL_DATA_DIR", str(data_dir))
    monkeypatch.delenv("DRAFT_COACH_DERIVED_DIR", raising=False)
    monkeypatch.delenv("CLOUD9_TEAM_ID", raising=False)

    settings_type = cast(Any, Settings)
    defaults = settings_type(_env_file=None)
    assert defaults.derived_dir == DEFAULT_DERIVED_DIR
    assert defaults.cloud9_team_id is None

    monkeypatch.setenv("DRAFT_COACH_DERIVED_DIR", str(tmp_path / "derived"))
    monkeypatch.setenv("CLOUD9_TEAM_ID", "47351")
    configured = Settings()
    assert configured.derived_dir == tmp_path / "derived"
    assert configured.cloud9_team_id == "47351"
