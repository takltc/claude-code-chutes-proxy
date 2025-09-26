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
        # Auto model-case fix (can trigger /v1/models discovery). Keep default on, but we'll cache results.
        self.auto_fix_model_case: bool = (os.environ.get("AUTO_FIX_MODEL_CASE", "1").lower() in ("1", "true", "yes"))
        # Whether to attempt pre-flight model-case discovery before streaming request
        # Defaults to off to avoid extra network RTT; we still retry on 404.
        self.auto_fix_model_case_preflight: bool = (
            os.environ.get("AUTO_FIX_MODEL_CASE_PREFLIGHT", "0").lower() in ("1", "true", "yes")
        )
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
        # Enable HTTP/2 to improve latency and throughput when supported by upstream.
        self.http2: bool = (os.environ.get("PROXY_HTTP2", "1").lower() in ("1", "true", "yes"))
        # Cache TTL (seconds) for /v1/models discovery used in case-fixing.
        self.model_discovery_ttl: int = int(os.environ.get("MODEL_DISCOVERY_TTL", "300"))
        # Optional: disable streaming tool-call parser to reduce per-chunk CPU.
        self.enable_stream_tool_parser: bool = (
            os.environ.get("ENABLE_STREAM_TOOL_PARSER", "0").lower() in ("1", "true", "yes")
        )
        # Persist discovered model list to disk (per upstream + auth fingerprint)
        self.model_discovery_persist: bool = (
            os.environ.get("MODEL_DISCOVERY_PERSIST", "1").lower() in ("1", "true", "yes")
        )
        # Path to models cache file (JSON). Default under home dir.
        default_cache_path = os.path.join(
            os.path.expanduser("~"),
            ".claude-code-chutes-proxy",
            "models_cache.json",
        )
        self.model_cache_file: str = os.environ.get("MODEL_CACHE_FILE", default_cache_path)
        self.chutes_max_tokens: int = max(1, int(os.environ.get("CHUTES_MAX_TOKENS", "128000")))
        # How much of the window to keep after compaction; defaults mirror Roo-Code heuristics
        try:
            self.chutes_token_buffer_ratio: float = float(os.environ.get("CHUTES_TOKEN_BUFFER_RATIO", str(0.85)))
        except Exception:
            self.chutes_token_buffer_ratio = 0.85
        try:
            self.chutes_tail_reserve: int = max(1, int(os.environ.get("CHUTES_TAIL_RESERVE", "6")))
        except Exception:
            self.chutes_tail_reserve = 6
        try:
            self.chutes_response_token_reserve: int = max(0, int(os.environ.get("CHUTES_RESPONSE_TOKEN_RESERVE", "4096")))
        except Exception:
            self.chutes_response_token_reserve = 4096
        try:
            self.chutes_min_context_tokens: int = max(512, int(os.environ.get("CHUTES_MIN_CONTEXT_TOKENS", "4096")))
        except Exception:
            self.chutes_min_context_tokens = 4096
        self.chutes_summary_model: Optional[str] = os.environ.get("CHUTES_SUMMARY_MODEL")
        try:
            self.chutes_summary_max_tokens: int = max(256, int(os.environ.get("CHUTES_SUMMARY_MAX_TOKENS", "1024")))
        except Exception:
            self.chutes_summary_max_tokens = 1024
        try:
            self.chutes_summary_keep_last: int = max(1, int(os.environ.get("CHUTES_SUMMARY_KEEP_LAST", "4")))
        except Exception:
            self.chutes_summary_keep_last = 4
        try:
            self.chutes_auto_condense_percent: int = max(10, int(os.environ.get("CHUTES_AUTO_CONDENSE_PERCENT", "100")))
        except Exception:
            self.chutes_auto_condense_percent = 100
        try:
            self.chutes_context_safety_tokens: int = max(0, int(os.environ.get("CHUTES_CONTEXT_SAFETY_TOKENS", "2048")))
        except Exception:
            self.chutes_context_safety_tokens = 2048

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
