import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import zstandard


@pytest.fixture
def write_zstd_feed(tmp_path: Path) -> Callable[[str, list[dict[str, Any]]], Path]:
    def write_feed(filename: str, envelopes: list[dict[str, Any]]) -> Path:
        path = tmp_path / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        with zstandard.open(path, "wt", encoding="utf-8") as file:
            for envelope in envelopes:
                file.write(json.dumps(envelope) + "\n")
        return path

    return write_feed
