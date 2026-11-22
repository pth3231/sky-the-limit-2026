from collections.abc import Mapping

from app.pipeline.meta import Champion
from app.pipeline.sources.oracles_elixir import (
    OE_POSITION_ROLES,
    _number,
    build_champion_id_map,
    resolve_champion_id,
)


def _champions() -> Mapping[str, Champion]:
    return {
        "Aatrox": Champion(character_id="grid-aatrox", name="Aatrox", games_played=1),
        "LeeSin": Champion(character_id="grid-leesin", name="Lee Sin", games_played=1),
    }


def test_position_roles_cover_oe_abbreviations() -> None:
    assert OE_POSITION_ROLES == {
        "top": "top",
        "jng": "jungle",
        "mid": "mid",
        "bot": "bot",
        "sup": "support",
    }


def test_resolve_champion_id_maps_names_to_grid_ids_with_fallback() -> None:
    name_to_id = build_champion_id_map(_champions())
    assert resolve_champion_id("Aatrox", name_to_id) == "grid-aatrox"
    # Normalization ignores spaces/case, so OE "Lee Sin" matches GRID "Lee Sin".
    assert resolve_champion_id("Lee Sin", name_to_id) == "grid-leesin"
    # Unknown champion falls back to its normalized key, never raising.
    assert resolve_champion_id("Smolder", name_to_id) == "smolder"


def test_number_coerces_blanks_to_zero() -> None:
    assert _number("12.5") == 12.5
    assert _number("") == 0.0
    assert _number(None) == 0.0
    assert _number("n/a") == 0.0
