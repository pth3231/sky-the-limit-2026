"""Join extracted draft actions with game metadata for downstream processing."""

import csv
import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from app.pipeline.drafts import (
    DraftAction,
    PlayerLine,
    extract_game_drafts,
    extract_game_lineups,
    read_final_series_state,
)
from app.pipeline.meta import load_games


class GameDraft(BaseModel):
    """A game's ordered draft actions and competition metadata."""

    model_config = ConfigDict(frozen=True)

    game_id: str
    series_id: str
    game_number: int
    patch: str
    finished: bool
    winner_team_id: str | None
    blue_team_id: str
    red_team_id: str
    actions: list[DraftAction]
    started_at: str = ""
    lineups: list[PlayerLine] = Field(default_factory=list)


def _parse_series(
    data_dir: Path, series: dict[str, str]
) -> tuple[dict[str, list[DraftAction]], dict[str, list[PlayerLine]]]:
    """Parse one series feed into its per-game draft actions and lineups.

    Runs in a worker process, so it must not touch the ``games`` metadata lookup.
    """
    feed_path = data_dir / series["file"]
    series_state = read_final_series_state(feed_path)
    return extract_game_drafts(series_state), extract_game_lineups(series_state)


def build_game_drafts(data_dir: Path, *, verbose: bool = False) -> list[GameDraft]:
    """Build game-level draft records from indexed series feeds and metadata."""
    games = load_games(data_dir)
    series_path = data_dir / "lol" / "series.csv"
    game_drafts: list[GameDraft] = []

    with series_path.open(newline="", encoding="utf-8-sig") as file:
        all_series = list(csv.DictReader(file))

    total = len(all_series)
    start = time.monotonic()
    with ProcessPoolExecutor(mp_context=multiprocessing.get_context("spawn")) as executor:
        futures = {
            executor.submit(_parse_series, data_dir, series): series for series in all_series
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            if verbose:
                elapsed = time.monotonic() - start
                average = elapsed / completed
                remaining = average * (total - completed)
                print(
                    f"Series {completed}/{total}: elapsed={elapsed:.1f}s ETA={remaining:.1f}s",
                    flush=True,
                )
            drafts, lineups = future.result()
            for game_id, actions in drafts.items():
                game = games.get(game_id)
                if game is None or not actions:
                    continue
                game_drafts.append(
                    GameDraft(
                        game_id=game.game_id,
                        series_id=game.series_id,
                        game_number=game.game_number,
                        patch=game.patch,
                        finished=game.finished,
                        winner_team_id=game.winner_team_id,
                        blue_team_id=game.blue_team_id,
                        red_team_id=game.red_team_id,
                        actions=actions,
                        started_at=game.started_at,
                        lineups=lineups.get(game_id, []),
                    )
                )

    return sorted(game_drafts, key=lambda game: (game.series_id, game.game_number))
