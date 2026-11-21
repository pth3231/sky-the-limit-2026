import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import app.model.train as train_module
from app.pipeline.dataset import GameDraft


def test_training_cli_uses_derived_settings_without_grid_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_dir = tmp_path / "configured-derived"
    drafts_path = tmp_path / "drafts.json"
    drafts_path.write_text(json.dumps([]), encoding="utf-8")
    captured: dict[str, Path] = {}

    def fake_train_model(games: list[GameDraft], output: Path, **kwargs: Any) -> SimpleNamespace:
        captured["output"] = output
        return SimpleNamespace(model_dump=lambda mode: {})

    monkeypatch.setattr(
        train_module,
        "DerivedSettings",
        lambda: SimpleNamespace(derived_dir=output_dir),
        raising=False,
    )
    monkeypatch.setattr(train_module, "train_model", fake_train_model)

    train_module.main(["--drafts", str(drafts_path)])

    assert captured["output"] == output_dir
