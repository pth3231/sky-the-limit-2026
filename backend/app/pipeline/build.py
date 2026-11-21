"""Build local draft and aggregate JSON artifacts from the GRID dataset."""

import argparse
import json
from pathlib import Path

from app.config import Settings
from app.pipeline.aggregates import PatchAggregate, compute_aggregates
from app.pipeline.dataset import GameDraft, build_game_drafts
from app.pipeline.ddragon import fetch_ddragon


def build_outputs(
    data_dir: Path,
    output_dir: Path,
    *,
    include_ddragon: bool = True,
    verbose: bool = False,
) -> tuple[list[GameDraft], dict[str, PatchAggregate]]:
    """Build game drafts and patch aggregates, writing both JSON artifacts."""
    game_drafts = build_game_drafts(data_dir, verbose=verbose)
    aggregates = compute_aggregates(game_drafts)
    output_dir.mkdir(parents=True, exist_ok=True)

    drafts_payload = [game.model_dump(mode="json") for game in game_drafts]
    aggregates_payload = {
        patch: aggregate.model_dump(mode="json") for patch, aggregate in aggregates.items()
    }
    (output_dir / "drafts.json").write_text(
        json.dumps(drafts_payload, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "aggregates.json").write_text(
        json.dumps(aggregates_payload, indent=2) + "\n",
        encoding="utf-8",
    )
    if include_ddragon:
        fetch_ddragon(output_dir)
    return game_drafts, aggregates


def main(argv: list[str] | None = None) -> None:
    """Run the data pipeline command-line entry point."""
    parser = argparse.ArgumentParser(description="Build Draft Coach local data artifacts.")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output directory (default: DRAFT_COACH_DERIVED_DIR or backend/.derived)",
    )
    parser.add_argument(
        "--no-ddragon",
        action="store_true",
        help="Skip the optional Data Dragon champion metadata download",
    )
    arguments = parser.parse_args(argv)

    settings = Settings()
    output_dir = arguments.out or settings.derived_dir
    game_drafts, aggregates = build_outputs(
        settings.data_dir,
        output_dir,
        include_ddragon=not arguments.no_ddragon,
        verbose=True,
    )
    print(
        f"Wrote {len(game_drafts)} game drafts and {len(aggregates)} patch aggregates "
        f"to {output_dir}"
    )


if __name__ == "__main__":
    main()
