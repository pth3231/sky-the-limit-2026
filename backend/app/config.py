"""Application settings and local GRID data directory resolution."""

import os
from collections.abc import Iterable, Mapping
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

DEFAULT_DERIVED_DIR = Path(__file__).resolve().parents[1] / ".derived"
BACKEND_ENV_FILE = Path(__file__).resolve().parents[1] / ".env"


class _AppSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=BACKEND_ENV_FILE,
        env_file_encoding="utf-8",
        populate_by_name=True,
    )


class DerivedSettings(_AppSettings):
    """Settings needed by model training without requiring the GRID data directory."""

    derived_dir: Path = Field(
        default=DEFAULT_DERIVED_DIR, validation_alias="DRAFT_COACH_DERIVED_DIR"
    )


def _require_meta(data_dir: Path) -> Path:
    resolved = data_dir.expanduser().resolve()
    if not resolved.is_dir() or not (resolved / "meta").is_dir():
        raise ValueError(f"GRID data directory must contain a meta/ directory: {resolved}")
    return resolved


def resolve_data_dir(env: Mapping[str, str], candidates: Iterable[Path]) -> Path:
    """Resolve a configured or candidate GRID data directory.

    The selected directory must contain the metadata index in its `meta/` subdirectory.
    """
    configured = env.get("STL_DATA_DIR")
    if configured:
        return _require_meta(Path(configured))

    for candidate in candidates:
        if candidate.exists():
            return _require_meta(candidate)

    raise FileNotFoundError(
        "GRID data directory not found; set STL_DATA_DIR or place "
        "stl-hackathon-2026-grid-starter beside the project."
    )


def _default_data_dir() -> Path:
    project_root = Path(__file__).resolve().parents[2]
    candidates = (project_root.parent / "stl-hackathon-2026-grid-starter",)
    return resolve_data_dir(os.environ, candidates)


class Settings(_AppSettings):
    """Validated application settings loaded from the backend environment file."""

    data_dir: Path = Field(default_factory=_default_data_dir, validation_alias="STL_DATA_DIR")
    derived_dir: Path = Field(
        default=DEFAULT_DERIVED_DIR, validation_alias="DRAFT_COACH_DERIVED_DIR"
    )
    cloud9_team_id: str | None = Field(default=None, validation_alias="CLOUD9_TEAM_ID")

    @field_validator("data_dir")
    @classmethod
    def data_dir_contains_meta(cls, value: Path) -> Path:
        """Resolve the configured directory and require its `meta/` index."""
        return _require_meta(value)
