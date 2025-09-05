from __future__ import annotations

import asyncio
from typing import Any, Dict, Optional

import httpx


class SchemaRegistry:
    def __init__(self) -> None:
        self.models: Dict[str, Dict[str, Any]] = {}
        self._loaded: bool = False
        self._last_fetch_ts: float | None = None
        self._min_interval: float = float(int(__import__('os').environ.get('DISCOVERY_MIN_INTERVAL', '300')))
        self._lock = asyncio.Lock()
        self._id_lower_map: Dict[str, str] = {}

    def get(self, model_id: str) -> Optional[Dict[str, Any]]:
        return self.models.get(model_id)

    async def discover(self, base_url: str, headers: Dict[str, str]) -> None:
        base = base_url.rstrip("/")
        try:
            async with self._lock:
                import time
                now = time.time()
                if self._last_fetch_ts and (now - self._last_fetch_ts) < self._min_interval:
                    return
                async with httpx.AsyncClient(timeout=httpx.Timeout(20.0)) as client:
                    res = await client.get(f"{base}/v1/models", headers=headers)
                    if res.status_code >= 400:
                        return
                    data = res.json()
                    items = []
                    if isinstance(data, dict):
                        if isinstance(data.get("data"), list):
                            items = data["data"]
                        elif isinstance(data.get("models"), list):
                            items = data["models"]
                    ids: list[str] = []
                    for it in items:
                        mid = it.get("id") if isinstance(it, dict) else (it if isinstance(it, str) else None)
                        if mid:
                            ids.append(mid)
                    # fetch details in parallel
                    async def fetch_detail(mid: str):
                        detail = None
                        try:
                            r = await client.get(f"{base}/v1/models/{mid}", headers=headers)
                            if r.status_code < 400:
                                detail = r.json()
                        except Exception:
                            detail = None
                        caps: Dict[str, Any] = {}
                        if isinstance(detail, dict):
                            # Try common capability keys
                            for key in ("capabilities", "caps", "features"):
                                if isinstance(detail.get(key), dict):
                                    caps.update(detail[key])
                            # Heuristics for vision/tooling
                            text = str(detail).lower()
                            if "tool" in text:
                                caps.setdefault("tools", True)
                            if "vision" in text or "image" in text:
                                caps.setdefault("vision", True)
                            if "reasoning" in text:
                                caps.setdefault("reasoning", True)
                        self.models[mid] = {"id": mid, "detail": detail, "caps": caps}
                    await asyncio.gather(*(fetch_detail(mid) for mid in ids))
                    # Build lower-case map
                    self._id_lower_map = {mid.lower(): mid for mid in self.models.keys()}
                    self._loaded = True
                    self._last_fetch_ts = now
        except Exception:
            return

    async def ensure(self, base_url: str, headers: Dict[str, str]) -> None:
        if not self._loaded:
            await self.discover(base_url, headers)

    async def ensure_model_entry(self, base_url: str, headers: Dict[str, str], model_id: str) -> None:
        """Ensure a registry entry exists; add heuristic if the provider has no per-model schema."""
        if not model_id:
            return
        if model_id in self.models:
            return
        # Attempt a refresh inside rate-limited window
        await self.discover(base_url, headers)
        if model_id in self.models:
            return
        caps: Dict[str, Any] = {}
        if not caps:
            caps.update({"tools": True})
        self.models[model_id] = {"id": model_id, "detail": None, "caps": caps}

    def resolve_case_variant(self, requested_model: str) -> Optional[str]:
        if not requested_model:
            return None
        lower = requested_model.lower()
        actual = self._id_lower_map.get(lower)
        if actual and actual != requested_model:
            return actual
        return None


registry = SchemaRegistry()
