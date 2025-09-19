import asyncio
import json
from typing import Any, Dict

import pytest

from app import main as main_mod


class _FakeResponse:
    status_code = 200

    def __init__(self, payload: Dict[str, Any]):
        self._payload = payload

    def json(self) -> Dict[str, Any]:
        return self._payload


class _FakeClient:
    def __init__(self, payload: Dict[str, Any]):
        self._payload = payload
        self.call_count = 0

    async def get(self, url: str, headers: Dict[str, str], timeout: Any) -> _FakeResponse:
        self.call_count += 1
        return _FakeResponse(self._payload)


def test_model_cache_refreshes_when_disk_entry_stale(monkeypatch, tmp_path):
    main_mod._MODEL_LIST_CACHE.clear()
    main_mod._MODEL_CASE_MAP_CACHE.clear()

    cache_path = tmp_path / "models_cache.json"

    monkeypatch.setattr(main_mod.settings, "model_discovery_persist", True)
    monkeypatch.setattr(main_mod.settings, "model_cache_file", str(cache_path))
    monkeypatch.setattr(main_mod.settings, "model_discovery_ttl", 60)

    key = main_mod._models_cache_key({})
    stale_entry = {
        "ids": ["zai-org/GLM-4.5-FP8"],
        "ids_lower": ["zai-org/glm-4.5-fp8"],
        "lower_map": {"zai-org/glm-4.5-fp8": "zai-org/GLM-4.5-FP8"},
        "ts": 12345,
        "base_url": main_mod.settings.chutes_base_url,
    }
    cache_path.write_text(json.dumps({key: stale_entry}))

    fake_payload = {"data": [{"id": "zai-org/GLM-4.5-Turbo"}]}
    fake_client = _FakeClient(fake_payload)
    monkeypatch.setattr(main_mod, "_get_httpx_client", lambda: fake_client)

    ids = asyncio.run(main_mod._get_model_ids({}))

    assert "zai-org/GLM-4.5-Turbo" in ids
    assert fake_client.call_count == 1

    cache_obj = json.loads(cache_path.read_text())
    entry = cache_obj[key]
    assert entry["lower_map"]["zai-org/glm-4.5-turbo"] == "zai-org/GLM-4.5-Turbo"
    assert entry["ts"] >= 1_000_000_000
    assert entry.get("ts_wall", entry["ts"]) >= 1_000_000_000

    cached_map = main_mod._MODEL_CASE_MAP_CACHE[key][1]
    assert cached_map["zai-org/glm-4.5-turbo"] == "zai-org/GLM-4.5-Turbo"

    main_mod._MODEL_LIST_CACHE.clear()
    main_mod._MODEL_CASE_MAP_CACHE.clear()
