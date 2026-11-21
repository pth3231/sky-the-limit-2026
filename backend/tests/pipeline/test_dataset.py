import csv
from pathlib import Path
from typing import Any

from app.pipeline.dataset import build_game_drafts


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_joins_known_games_and_skips_unknown_or_undrafted_games(
    tmp_path: Path,
    write_zstd_feed: Any,
) -> None:
    _write_csv(
        tmp_path / "meta" / "games.csv",
        [
            {
                "title": "lol",
                "series_id": "series-1",
                "game_id": "game-known",
                "game_number": "1",
                "patch": "15.10",
                "started_at": "2026-10-10T12:00:00Z",
                "finished": "true",
                "winner_team_id": "team-blue",
                "team1_id": "team-blue",
                "team1_side": "blue",
                "team2_id": "team-red",
                "team2_side": "red",
            },
            {
                "title": "lol",
                "series_id": "series-1",
                "game_id": "game-empty",
                "game_number": "2",
                "patch": "15.10",
                "finished": "false",
                "winner_team_id": "",
                "team1_id": "team-blue",
                "team1_side": "blue",
                "team2_id": "team-red",
                "team2_side": "red",
            },
        ],
    )
    _write_csv(
        tmp_path / "lol" / "series.csv",
        [{"series_id": "series-1", "file": "lol/events_series-1_grid.jsonl.zst"}],
    )
    write_zstd_feed(
        "lol/events_series-1_grid.jsonl.zst",
        [
            {
                "events": [
                    {
                        "seriesState": {
                            "games": [
                                {
                                    "id": "game-known",
                                    "teams": [
                                        {
                                            "id": "team-blue",
                                            "players": [
                                                {"character": {"id": f"champ-{index}"}}
                                                for index in range(5)
                                            ],
                                        }
                                    ],
                                    "draftActions": [
                                        {
                                            "sequenceNumber": 1,
                                            "type": "pick",
                                            "drafter": {"id": "team-blue"},
                                            "draftable": {"id": "champ-1", "name": "Aster"},
                                        }
                                    ],
                                },
                                {"id": "game-empty", "draftActions": []},
                                {
                                    "id": "game-not-in-meta",
                                    "draftActions": [
                                        {
                                            "sequenceNumber": 2,
                                            "type": "ban",
                                            "drafter": {"id": "team-red"},
                                            "draftable": {"id": "champ-2", "name": "Beryl"},
                                        }
                                    ],
                                },
                            ]
                        }
                    }
                ]
            }
        ],
    )

    drafts = build_game_drafts(tmp_path)

    assert len(drafts) == 1
    assert drafts[0].game_id == "game-known"
    assert drafts[0].series_id == "series-1"
    assert drafts[0].patch == "15.10"
    assert drafts[0].started_at == "2026-10-10T12:00:00Z"
    assert drafts[0].blue_team_id == "team-blue"
    assert drafts[0].red_team_id == "team-red"
    assert drafts[0].winner_team_id == "team-blue"
    assert drafts[0].actions[0].champion_id == "champ-1"
    assert len(drafts[0].lineups) == 5
    assert drafts[0].lineups[0].role is None
