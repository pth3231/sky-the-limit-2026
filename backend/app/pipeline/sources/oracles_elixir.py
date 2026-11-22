"""Parse Oracle's Elixir match CSVs into the pipeline's GameDraft shape."""

from collections.abc import Mapping

from app.pipeline.ddragon import normalize_champion_name
from app.pipeline.drafts import Role
from app.pipeline.meta import Champion

OE_POSITION_ROLES: dict[str, Role] = {
    "top": "top",
    "jng": "jungle",
    "mid": "mid",
    "bot": "bot",
    "sup": "support",
}


def _number(value: object) -> float:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0.0


def build_champion_id_map(champions: Mapping[str, Champion]) -> dict[str, str]:
    """Map normalized champion display names to GRID character ids."""
    return {
        normalize_champion_name(champion.name): champion.character_id
        for champion in champions.values()
    }


def resolve_champion_id(name: str, name_to_id: Mapping[str, str]) -> str:
    """Resolve an OE champion name to a GRID id, falling back to its normalized key."""
    normalized = normalize_champion_name(name)
    return name_to_id.get(normalized, normalized)
