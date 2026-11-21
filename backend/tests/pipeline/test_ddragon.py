import json
from pathlib import Path
from typing import Any

import app.pipeline.ddragon as ddragon


def test_parses_champion_tags_and_info_by_normalized_name() -> None:
    champions = ddragon.parse_champion_json(
        {
            "data": {
                "KSante": {
                    "name": "K'Sante",
                    "tags": ["Tank", "Fighter"],
                    "info": {"attack": 8, "defense": 6, "magic": 2, "difficulty": 7},
                }
            }
        }
    )

    assert champions["ksante"].tags == ("Tank", "Fighter")
    assert champions["ksante"].info["attack"] == 8
    assert ddragon.normalize_champion_name("Nunu & Willump") == "nunuwillump"


def test_parses_versioned_portrait_urls_and_known_aliases() -> None:
    champions = ddragon.parse_champion_json(
        {
            "data": {
                "Aatrox": {
                    "name": "Aatrox",
                    "tags": ["Fighter"],
                    "info": {},
                    "image": {"full": "Aatrox.png"},
                }
            }
        },
        version="16.1.1",
    )

    assert champions["aatrox"].portrait_url == (
        "https://ddragon.leagueoflegends.com/cdn/16.1.1/img/champion/Aatrox.png"
    )
    assert ddragon.normalize_champion_name("Nunu") == ddragon.normalize_champion_name(
        "Nunu & Willump"
    )


def test_fetches_latest_data_and_writes_cache(tmp_path: Path, monkeypatch: Any) -> None:
    responses = [
        ["16.1.1"],
        {
            "data": {
                "Aatrox": {
                    "name": "Aatrox",
                    "tags": ["Fighter"],
                    "info": {"attack": 8},
                    "image": {"full": "Aatrox.png"},
                }
            }
        },
    ]
    urls: list[str] = []

    class Response:
        def __enter__(self) -> "Response":
            return self

        def __exit__(self, *_: object) -> None:
            return None

        def read(self) -> bytes:
            return json.dumps(responses.pop(0)).encode()

    def fake_urlopen(url: str, timeout: float) -> Response:
        urls.append(url)
        assert timeout == 1.5
        return Response()

    monkeypatch.setattr(ddragon.urllib.request, "urlopen", fake_urlopen)

    champions = ddragon.fetch_ddragon(tmp_path, timeout=1.5)

    assert urls == [
        ddragon.VERSIONS_URL,
        "https://ddragon.leagueoflegends.com/cdn/16.1.1/data/en_US/champion.json",
    ]
    assert champions["aatrox"].tags == ("Fighter",)
    assert json.loads((tmp_path / "ddragon.json").read_text()) == {
        "aatrox": {
            "tags": ["Fighter"],
            "info": {"attack": 8.0},
            "portrait_url": "https://ddragon.leagueoflegends.com/cdn/16.1.1/img/champion/Aatrox.png",
        }
    }


def test_fetch_failure_falls_back_to_empty_metadata(tmp_path: Path, monkeypatch: Any) -> None:
    def fail_urlopen(_: str, timeout: float) -> None:
        raise OSError("offline")

    monkeypatch.setattr(ddragon.urllib.request, "urlopen", fail_urlopen)

    assert ddragon.fetch_ddragon(tmp_path) == {}


def test_load_portraits_refreshes_legacy_cache_and_works_offline(
    tmp_path: Path, monkeypatch: Any
) -> None:
    cache = tmp_path / "ddragon.json"
    cache.write_text(json.dumps({"aatrox": {"tags": [], "info": {}}}))
    calls: list[Path] = []

    def refresh(path: Path, timeout: float = 3.0) -> dict[str, ddragon.ChampionTraits]:
        calls.append(path)
        return {
            "aatrox": ddragon.ChampionTraits(
                tags=(), info={}, portrait_url="https://example.test/aatrox.png"
            )
        }

    monkeypatch.setattr(ddragon, "fetch_ddragon", refresh)
    assert ddragon.load_portrait_urls(cache) == {"aatrox": "https://example.test/aatrox.png"}
    assert calls == [tmp_path]

    monkeypatch.setattr(ddragon, "fetch_ddragon", lambda *args, **kwargs: {})
    assert ddragon.load_portrait_urls(cache) == {}
