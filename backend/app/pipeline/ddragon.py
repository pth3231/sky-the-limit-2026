"""Best-effort download and normalization of Data Dragon champion metadata."""

import json
import re
import urllib.request
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from urllib.parse import quote

from pydantic import BaseModel, ConfigDict

VERSIONS_URL = "https://ddragon.leagueoflegends.com/api/versions.json"
_CACHE_NAME = "ddragon.json"


class ChampionTraits(BaseModel):
    """Data Dragon traits and the official champion portrait URL, when cached."""

    model_config = ConfigDict(frozen=True)

    tags: tuple[str, ...]
    info: dict[str, float]
    portrait_url: str | None = None


_NAME_ALIASES = {
    "monkeyking": "wukong",
    "nunu": "nunuwillump",
}


def normalize_champion_name(name: str) -> str:
    """Remove spaces and punctuation so GRID and Data Dragon names can be joined."""
    normalized = re.sub(r"[^a-z0-9]", "", name.casefold())
    return _NAME_ALIASES.get(normalized, normalized)


def parse_champion_json(
    payload: Mapping[str, Any], version: str | None = None
) -> dict[str, ChampionTraits]:
    """Parse Data Dragon's champion JSON into normalized tags and numeric info."""
    champion_data = payload.get("data")
    if not isinstance(champion_data, dict):
        return {}

    traits: dict[str, ChampionTraits] = {}
    for champion in champion_data.values():
        if not isinstance(champion, dict) or not isinstance(champion.get("name"), str):
            continue
        key = normalize_champion_name(champion["name"])
        tags = champion.get("tags", [])
        info = champion.get("info", {})
        if not key or not isinstance(tags, list) or not isinstance(info, dict):
            continue
        numeric_info = {
            name: float(value)
            for name, value in info.items()
            if isinstance(name, str)
            and isinstance(value, (int, float))
            and not isinstance(value, bool)
        }
        image = champion.get("image")
        filename = image.get("full") if isinstance(image, dict) else None
        portrait_url = None
        if version and isinstance(filename, str) and filename:
            portrait_url = (
                "https://ddragon.leagueoflegends.com/cdn/"
                f"{quote(version, safe='')}/img/champion/{quote(filename, safe='')}"
            )
        traits[key] = ChampionTraits(
            tags=tuple(tag for tag in tags if isinstance(tag, str)),
            info=numeric_info,
            portrait_url=portrait_url,
        )
    return traits


def _read_cache(path: Path) -> dict[str, ChampionTraits]:
    try:
        cached = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(cached, dict):
        return {}

    result: dict[str, ChampionTraits] = {}
    for name, value in cached.items():
        if not isinstance(name, str) or not isinstance(value, dict):
            continue
        try:
            result[name] = ChampionTraits.model_validate(value)
        except ValueError:
            continue
    return result


def load_portrait_urls(path: Path) -> dict[str, str]:
    """Read portraits, refreshing an absent or pre-portrait cache when possible."""
    champions = _read_cache(path)
    if not any(champion.portrait_url for champion in champions.values()):
        champions = fetch_ddragon(path.parent, timeout=1.0)
    return {
        name: champion.portrait_url
        for name, champion in champions.items()
        if champion.portrait_url
    }


def fetch_ddragon(out: Path, timeout: float = 3.0) -> dict[str, ChampionTraits]:
    """Download latest champion traits; use the local cache or empty data if unavailable."""
    cache_path = out / _CACHE_NAME
    try:
        with urllib.request.urlopen(VERSIONS_URL, timeout=timeout) as response:
            versions = json.load(response)
        if not isinstance(versions, list) or not versions or not isinstance(versions[0], str):
            return _read_cache(cache_path)

        url = f"https://ddragon.leagueoflegends.com/cdn/{versions[0]}/data/en_US/champion.json"
        with urllib.request.urlopen(url, timeout=timeout) as response:
            champions = parse_champion_json(json.load(response), version=versions[0])
        if not champions:
            return _read_cache(cache_path)

        out.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(
            json.dumps(
                {name: champion.model_dump(mode="json") for name, champion in champions.items()},
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        return champions
    except (OSError, TypeError, ValueError):
        return _read_cache(cache_path)
