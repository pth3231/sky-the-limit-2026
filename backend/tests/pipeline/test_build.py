import csv
import json
from pathlib import Path
from typing import Any

import app.pipeline.build as build_module
from app.pipeline.build import main


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_build_writes_drafts_and_aggregates_json(
    tmp_path: Path,
    monkeypatch: Any,
    write_zstd_feed: Any,
) -> None:
    data_dir = tmp_path / "data"
    _write_csv(
        data_dir / "meta" / "games.csv",
        [
            {
                "title": "lol",
                "series_id": "series-1",
                "game_id": "game-1",
                "game_number": "1",
                "patch": "15.10",
                "finished": "true",
                "winner_team_id": "blue",
                "team1_id": "blue",
                "team1_side": "blue",
                "team2_id": "red",
                "team2_side": "red",
            }
        ],
    )
    _write_csv(
        data_dir / "lol" / "series.csv",
        [{"series_id": "series-1", "file": "lol/events_series-1_grid.jsonl.zst"}],
    )
    write_zstd_feed(
        "data/lol/events_series-1_grid.jsonl.zst",
        [
            {
                "events": [
                    {
                        "seriesState": {
                            "games": [
                                {
                                    "id": "game-1",
                                    "draftActions": [
                                        {
                                            "sequenceNumber": 1,
                                            "type": "pick",
                                            "drafter": {"id": "blue"},
                                            "draftable": {"id": "champ-1", "name": "Aster"},
                                        }
                                    ],
                                }
                            ]
                        }
                    }
                ]
            }
        ],
    )
    monkeypatch.setenv("STL_DATA_DIR", str(data_dir))
    output_dir = tmp_path / "output"

    main(["--no-ddragon", "--out", str(output_dir)])

    drafts = json.loads((output_dir / "drafts.json").read_text(encoding="utf-8"))
    aggregates = json.loads((output_dir / "aggregates.json").read_text(encoding="utf-8"))
    assert not (output_dir / "ddragon.json").exists()
    assert drafts[0]["game_id"] == "game-1"
    assert drafts[0]["actions"][0]["champion_id"] == "champ-1"
    assert aggregates["15.10"]["champions"][0]["picks"] == 1
    assert aggregates["15.10"]["sides"]["blue_wins"] == 1


def test_configured_derived_dir_is_the_default_output(tmp_path: Path, monkeypatch: Any) -> None:
    data_dir = tmp_path / "data"
    (data_dir / "meta").mkdir(parents=True)
    output_dir = tmp_path / "configured-derived"
    monkeypatch.setenv("STL_DATA_DIR", str(data_dir))
    monkeypatch.setenv("DRAFT_COACH_DERIVED_DIR", str(output_dir))
    calls: list[tuple[Path, Path, bool]] = []

    monkeypatch.setattr(
        build_module,
        "build_outputs",
        lambda source, destination, *, include_ddragon=True, verbose=False: (
            calls.append((source, destination, include_ddragon)) or ([], {})
        ),
    )

    main([])

    assert calls == [(data_dir, output_dir, True)]
