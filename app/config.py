import json
import os
from typing import Dict, Optional


class Settings:
    def __init__(self) -> None:
        self.chutes_base_url: str = os.environ.get("CHUTES_BASE_URL", "https://llm.chutes.ai")
        self.chutes_api_key: Optional[str] = os.environ.get("CHUTES_API_KEY")
        # MODEL_MAP expects a JSON object string mapping Anthropic model names → backend model names
        model_map_raw = os.environ.get("MODEL_MAP", "{}")
        try:
            self.model_map: Dict[str, str] = json.loads(model_map_raw)
        except Exception:
            self.model_map = {}
        self.debug: bool = (os.environ.get("DEBUG_PROXY", "").lower() in ("1", "true", "yes"))
        self.auto_fix_model_case: bool = (os.environ.get("AUTO_FIX_MODEL_CASE", "1").lower() in ("1", "true", "yes"))
        self.backoff_on_429: bool = (os.environ.get("PROXY_BACKOFF_ON_429", "1").lower() in ("1", "true", "yes"))
        self.max_retry_on_429: int = int(os.environ.get("PROXY_MAX_RETRY_ON_429", "1"))
        self.max_retry_after_seconds: float = float(os.environ.get("PROXY_MAX_RETRY_AFTER", "2"))
        # Auth header style for upstream (x-api-key | authorization | both)
        self.chutes_auth_style: str = os.environ.get("CHUTES_AUTH_STYLE", "both").strip().lower()
        # Optional tool name mapping (global or per-model). Example:
        # {"some/model": {"list_repo_files": "ls", "read_file": "readFile"}}
        tool_name_map_raw = os.environ.get("TOOL_NAME_MAP", "{}")
        try:
            self.tool_name_map = json.loads(tool_name_map_raw)
        except Exception:
            self.tool_name_map = {}

    def map_model(self, anthropic_model: str) -> str:
        return self.model_map.get(anthropic_model, anthropic_model)

    def map_tool_name(self, model: Optional[str], name: Optional[str]) -> Optional[str]:
        if not name:
            return name
        mapping = None
        # Prefer per-model mapping if present
        if isinstance(self.tool_name_map, dict) and model and isinstance(self.tool_name_map.get(model), dict):
            mapping = self.tool_name_map.get(model)
        elif isinstance(self.tool_name_map, dict):
            mapping = self.tool_name_map
        if isinstance(mapping, dict) and name in mapping:
            return mapping[name]
        return name


settings = Settings()
