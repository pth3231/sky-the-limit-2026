from typing import Any

import pytest

from app.pipeline.drafts import extract_game_drafts, read_final_series_state


def test_reads_last_event_snapshot_from_compressed_feed(
    write_zstd_feed: Any,
) -> None:
    path = write_zstd_feed(
        "events.jsonl.zst",
        [
            {"events": [{"type": "feed-started"}]},
            {"events": [{"seriesState": {"games": [{"id": "old"}]}}]},
            {"events": [{"seriesState": {"games": [{"id": "final"}]}}]},
            {"events": [{"type": "game-ended"}]},
        ],
    )

    state = read_final_series_state(path)

    assert state["games"][0]["id"] == "final"


def test_missing_series_snapshot_fails_clearly(
    write_zstd_feed: Any,
) -> None:
    path = write_zstd_feed("empty.jsonl.zst", [{"events": [{"type": "feed-started"}]}])

    with pytest.raises(ValueError, match="seriesState"):
        read_final_series_state(path)


def test_extracts_ordered_draft_actions_per_game() -> None:
    state: dict[str, Any] = {
        "games": [
            {
                "id": "game-1",
                "draftActions": [
                    {
                        "sequenceNumber": 2,
                        "type": "pick",
                        "drafter": {"id": "team-2"},
                        "draftable": {"id": "champ-2", "name": "Beryl"},
                    },
                    {
                        "sequenceNumber": 1,
                        "type": "ban",
                        "drafter": {"id": "team-1"},
                        "draftable": {"id": "champ-1", "name": "Aster"},
                    },
                ],
            },
            {"id": "game-2", "draftActions": []},
        ]
    }

    drafts = extract_game_drafts(state)

    assert [action.sequence_number for action in drafts["game-1"]] == [1, 2]
    assert drafts["game-1"][0].action_type == "ban"
    assert drafts["game-1"][0].team_id == "team-1"
    assert drafts["game-1"][0].champion_id == "champ-1"
    assert drafts["game-1"][0].champion_name == "Aster"
    assert drafts["game-2"] == []
