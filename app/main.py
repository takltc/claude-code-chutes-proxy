from __future__ import annotations

import asyncio
import json
import re
import sys
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple
import uuid

import httpx
import os
from fastapi import FastAPI, Header, HTTPException, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse
from collections import deque

from .config import settings
from .schemas.anthropic import (
    ErrorResponse,
    MessageResponse,
    MessagesRequest,
)
from .transform import (
    anthropic_to_openai_payload,
    map_finish_reason,
    openai_to_anthropic_response,
    choose_tool_call_parser,
    parse_mcp_tool_markup,
)


app = FastAPI(title="Claude-to-Chutes Proxy")

# Shared HTTP client (HTTP/1.1 + optional HTTP/2) with connection pooling
_HTTPX_CLIENT: Optional[httpx.AsyncClient] = None


def _get_httpx_client() -> httpx.AsyncClient:
    global _HTTPX_CLIENT
    if _HTTPX_CLIENT is None:
        limits = httpx.Limits(
            max_connections=100,
            max_keepalive_connections=20,
            keepalive_expiry=30.0,
        )
        http2_flag = bool(getattr(settings, "http2", True))
        try:
            _HTTPX_CLIENT = httpx.AsyncClient(
                http2=http2_flag,
                limits=limits,
            )
        except ImportError:
            # If http2 extras not installed, gracefully fall back to HTTP/1.1
            _HTTPX_CLIENT = httpx.AsyncClient(
                http2=False,
                limits=limits,
            )
    return _HTTPX_CLIENT
_RECENT = deque(maxlen=64)


def _auth_headers(x_api_key: str | None = None, authorization: str | None = None) -> Dict[str, str]:
    headers: Dict[str, str] = {}
    # Build upstream auth headers according to configured style.
    style = (settings.chutes_auth_style or "both").lower()
    # Prefer explicit env var, else inbound x-api-key, else try bearer from inbound Authorization
    api_key: str | None = settings.chutes_api_key or x_api_key
    bearer_inbound: str | None = authorization

    def set_x_api_key(val: str) -> None:
        if val:
            headers["x-api-key"] = val

    def set_authorization(val: str) -> None:
        if not val:
            return
        if val.lower().startswith("bearer "):
            headers["Authorization"] = val
        else:
            headers["Authorization"] = f"Bearer {val}"

    if style == "x-api-key":
        # Only x-api-key
        if api_key:
            set_x_api_key(api_key)
        elif bearer_inbound:
            # try extracting token from inbound Authorization
            token = bearer_inbound.split(" ", 1)[1] if " " in bearer_inbound else bearer_inbound
            set_x_api_key(token)
    elif style == "authorization":
        # Only Authorization
        if bearer_inbound:
            set_authorization(bearer_inbound)
        elif api_key:
            set_authorization(api_key)
    else:
        # both (default): include as many as we can
        if api_key:
            set_x_api_key(api_key)
            set_authorization(api_key)
        else:
            if bearer_inbound:
                set_authorization(bearer_inbound)
            if x_api_key:
                set_x_api_key(x_api_key)

    return headers


def _strip_thinking_suffix(model: Optional[str]) -> tuple[Optional[str], bool]:
    """Strip trailing ":THINKING" (case-insensitive) from model id.

    Returns (new_model, enabled) where enabled indicates that THINKING mode was requested.
    """
    if not model:
        return model, False
    try:
        s = str(model).strip()
        # support :thinking / :think / :reason / :reasoning suffixes (case-insensitive)
        m = re.search(r":(thinking|think|reason|reasoning)\s*$", s, flags=re.IGNORECASE)
        if m:
            base = re.sub(r":(thinking|think|reason|reasoning)\s*$", "", s, flags=re.IGNORECASE)
            return base, True
    except Exception:
        ...
    return model, False


def _anthropic_error_for_status(status: int, message: str) -> Dict[str, Any]:
    t = "api_error"
    if status == 400:
        t = "invalid_request_error"
    elif status == 401:
        t = "authentication_error"
    elif status == 403:
        t = "permission_error"
    elif status == 404:
        t = "not_found_error"
    elif status == 429:
        t = "rate_limit_error"
    elif 500 <= status < 600:
        t = "api_error"
    return ErrorResponse(error={"type": t, "message": message}).model_dump()


# simple in-memory cache for model resolution
_MODEL_RESOLVE_CACHE: Dict[str, str] = {}
# Cache for provider model ids to avoid frequent /v1/models calls
_MODEL_LIST_CACHE: Dict[str, Tuple[float, List[str]]] = {}
# Cache of lowercase → canonical id maps per upstream/auth fingerprint
_MODEL_CASE_MAP_CACHE: Dict[str, Tuple[float, Dict[str, str]]] = {}
_MODEL_CACHE_LOCK = asyncio.Lock()


def _models_cache_key(headers: Dict[str, str]) -> str:
    # Keyed by base_url and a light fingerprint of auth headers
    auth = headers.get("Authorization") or headers.get("x-api-key") or settings.chutes_api_key or ""
    # avoid storing secrets; just length and last 6 chars as a weak fingerprint
    sig = f"len:{len(auth)}:{auth[-6:] if auth else ''}"
    return f"{settings.chutes_base_url.rstrip('/')}//{sig}"


async def _get_model_ids(headers: Dict[str, str]) -> Optional[List[str]]:
    key = _models_cache_key(headers)
    now = asyncio.get_event_loop().time()
    ttl = max(5, int(getattr(settings, "model_discovery_ttl", 300)))
    # In-memory cache first
    if key in _MODEL_LIST_CACHE:
        ts, ids = _MODEL_LIST_CACHE[key]
        if now - ts < ttl:
            return ids
    # Persistent cache (disk) second
    if settings.model_discovery_persist:
        try:
            # Load JSON file and pick current key
            if os.path.exists(settings.model_cache_file):
                with open(settings.model_cache_file, "r", encoding="utf-8") as f:
                    obj = json.load(f)
                ent = (obj or {}).get(key)
                if ent and isinstance(ent.get("ids"), list):
                    ids = [str(x) for x in ent.get("ids")]
                    # Populate case map cache if present
                    lmap = ent.get("lower_map") if isinstance(ent, dict) else None
                    if isinstance(lmap, dict):
                        _MODEL_CASE_MAP_CACHE[key] = (now, {str(k): str(v) for k, v in lmap.items()})
                    # Populate in-memory cache to avoid future file I/O
                    _MODEL_LIST_CACHE[key] = (now, ids)
                    return ids
        except Exception:
            ...
    url = f"{settings.chutes_base_url.rstrip('/')}/v1/models"
    client = _get_httpx_client()
    try:
        resp = await client.get(url, headers=headers, timeout=httpx.Timeout(30.0))
        if resp.status_code >= 400:
            return None
        data = resp.json()
    except Exception:
        return None
    items: List[Any] = []
    if isinstance(data, dict):
        if isinstance(data.get("data"), list):
            items = data["data"]
        elif isinstance(data.get("models"), list):
            items = data["models"]
    ids: List[str] = []
    for it in items:
        if isinstance(it, dict) and it.get("id"):
            ids.append(str(it["id"]))
        elif isinstance(it, str):
            ids.append(it)
    _MODEL_LIST_CACHE[key] = (now, ids)
    # Build and cache lowercase → canonical map
    lower_map: Dict[str, str] = {}
    for mid in ids:
        try:
            if isinstance(mid, str):
                norm = str(mid).strip().lower()
                lower_map[norm] = mid
        except Exception:
            ...
    _MODEL_CASE_MAP_CACHE[key] = (now, lower_map)
    # Persist to disk
    if settings.model_discovery_persist:
        try:
            os.makedirs(os.path.dirname(settings.model_cache_file), exist_ok=True)
            async with _MODEL_CACHE_LOCK:
                obj: Dict[str, Any] = {}
                if os.path.exists(settings.model_cache_file):
                    try:
                        with open(settings.model_cache_file, "r", encoding="utf-8") as f:
                            obj = json.load(f) or {}
                    except Exception:
                        obj = {}
                obj[key] = {
                    "ids": ids,               # canonical ids as returned by provider
                    "ids_lower": [i.lower() for i in ids],  # purely lowercased list for quick contains
                    "lower_map": lower_map,    # lowercase → canonical
                    "ts": int(now),
                    "base_url": settings.chutes_base_url,
                }
                with open(settings.model_cache_file, "w", encoding="utf-8") as f:
                    json.dump(obj, f, ensure_ascii=False, indent=2)
        except Exception:
            ...
    return ids


async def _get_model_case_map(headers: Dict[str, str]) -> Optional[Dict[str, str]]:
    key = _models_cache_key(headers)
    now = asyncio.get_event_loop().time()
    ttl = max(5, int(getattr(settings, "model_discovery_ttl", 300)))
    # in-memory
    if key in _MODEL_CASE_MAP_CACHE:
        ts, mp = _MODEL_CASE_MAP_CACHE[key]
        if now - ts < ttl:
            return mp
    # disk
    if settings.model_discovery_persist and os.path.exists(settings.model_cache_file):
        try:
            with open(settings.model_cache_file, "r", encoding="utf-8") as f:
                obj = json.load(f) or {}
            ent = obj.get(key)
            if isinstance(ent, dict) and isinstance(ent.get("lower_map"), dict):
                mp = {str(k): str(v) for k, v in ent.get("lower_map").items()}
                _MODEL_CASE_MAP_CACHE[key] = (now, mp)
                return mp
        except Exception:
            ...
    # Build from ids if available
    ids = await _get_model_ids(headers)
    if ids:
        mp: Dict[str, str] = {}
        for mid in ids:
            try:
                mp[mid.lower()] = mid
            except Exception:
                ...
        _MODEL_CASE_MAP_CACHE[key] = (now, mp)
        return mp
    return None


async def _try_resolve_model(
    requested_model: str, headers: Dict[str, str]
) -> str | None:
    # Use cache first
    if requested_model in _MODEL_RESOLVE_CACHE:
        return _MODEL_RESOLVE_CACHE[requested_model]
    ids = await _get_model_ids(headers) or []
    # exact match only (preserve case)
    if requested_model in ids:
        _MODEL_RESOLVE_CACHE[requested_model] = requested_model
        return requested_model
    return None


def _guess_case_for_known_models(requested_model: str) -> str | None:
    """Best-effort local guess for canonical casing of some known providers.

    Used only as a last-resort when provider discovery (/v1/models) is unavailable
    or does not include the requested model id.
    """
    try:
        s = (requested_model or "").strip()
        if not s or "/" not in s:
            return None
        vendor, name = s.split("/", 1)
        vlow = vendor.lower()
        nlow = name.strip().lower()
        # Moonshot Kimi K2 family commonly requires specific title-casing
        if vlow == "moonshotai" and (nlow.startswith("kimi-k2-") or nlow == "kimi-k2"):
            parts = [p for p in nlow.split("-") if p != ""]
            fixed: list[str] = []
            for p in parts:
                if p == "kimi":
                    fixed.append("Kimi")
                elif p == "k2":
                    fixed.append("K2")
                elif p.isdigit():
                    fixed.append(p)
                else:
                    fixed.append(p[:1].upper() + p[1:].lower())
            return f"{vendor}/{'-'.join(fixed)}"
    except Exception:
        return None
    return None


async def _resolve_case_variant(
    requested_model: str, headers: Dict[str, str]
) -> str | None:
    """Resolve canonical model id using cached lowercase→canonical map without network calls.

    Falls back to prior heuristic only if cache is unavailable.
    """
    mp = await _get_model_case_map(headers)
    if mp:
        # direct lowercase map
        req_low = (requested_model or "").strip().lower()
        val = mp.get(req_low)
        if val:
            return val
        # Heuristic: provider canonical ids may include a vendor prefix, e.g. "anthropic/<id>".
        # If the requested id lacks the prefix, try suffix-matching against known ids.
        try:
            # last segment after a vendor prefix (if any)
            last = req_low.split("/", 1)[-1]
            if last:
                # Prefer an exact suffix match "*/<last>"
                for lk, canon in mp.items():
                    try:
                        if isinstance(lk, str) and lk.endswith("/" + last):
                            return canon
                    except Exception:
                        ...
                # As a weaker fallback, accept exact key equality without prefix (when provider publishes plain ids)
                for lk, canon in mp.items():
                    try:
                        if isinstance(lk, str) and lk == last:
                            return canon
                    except Exception:
                        ...
        except Exception:
            ...

    # As a broader fallback, consult full provider model list and perform fuzzy resolution
    try:
        ids = await _get_model_ids(headers) or []
        if ids:
            req_low = (requested_model or "").strip().lower()
            last = req_low.split("/", 1)[-1]
            # 1) Exact last-segment match against any canonical id's last segment
            for canon in ids:
                try:
                    if not isinstance(canon, str):
                        continue
                    cl = canon.strip().lower()
                    if cl.split("/", 1)[-1] == last:
                        return canon
                except Exception:
                    ...
            # 2) If request has a date suffix like -YYYYMMDD, drop it and try again
            import re as _re
            m = _re.search(r"^(.*?)-(\d{8})$", last)
            base = m.group(1) if m else last
            if base and base != last:
                for canon in ids:
                    try:
                        if not isinstance(canon, str):
                            continue
                        cl = canon.strip().lower()
                        if cl.split("/", 1)[-1] == base:
                            return canon
                        if cl.endswith("/" + base + "-latest"):
                            return canon
                        if cl.endswith("/" + base):
                            return canon
                    except Exception:
                        ...
            # 3) Prefix match (e.g., base-latest, base-2024xxxx)
            if base:
                for canon in ids:
                    try:
                        if not isinstance(canon, str):
                            continue
                        cl = canon.strip().lower()
                        tail = cl.split("/", 1)[-1]
                        if tail.startswith(base):
                            return canon
                    except Exception:
                        ...
    except Exception:
        ...
    # Last-resort heuristic
    try:
        guessed = _guess_case_for_known_models(requested_model)
        if guessed and guessed != requested_model:
            return guessed
    except Exception:
        ...
    return None


@app.post("/v1/messages")
async def messages(
    request: Request,
    anthropic_version: str | None = Header(default=None, alias="anthropic-version"),
    x_api_key: str | None = Header(default=None, alias="x-api-key"),
    authorization: str | None = Header(default=None, alias="authorization"),
    x_enable_thinking: str | None = Header(default=None, alias="X-Enable-Thinking"),
):
    _rec: Dict[str, Any] = {"phase": "start"}
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    # Validate minimal schema
    try:
        parsed = MessagesRequest.model_validate(body)
    except Exception as e:
        return JSONResponse(
            status_code=400,
            content=ErrorResponse(error={"type": "invalid_request_error", "message": str(e)}).model_dump(),
        )
    _rec["request_model"] = parsed.model

    # Capture original model and detect :thinking-like suffix before any mapping
    _orig_model = parsed.model
    _orig_base, _orig_en = _strip_thinking_suffix(_orig_model)
    think_trace = {
        "model_in": _orig_model,
        "orig_suffix": bool(_orig_en),
        "payload_suffix": False,
        "body_thinking": False,
        "body_reasoning": False,
        "inbound_header": False,
        "final_enabled": False,
    }

    # Build OpenAI payload for Chutes（OpenAI 侧仅使用 sglang 协议模型校验/构造）
    oai_payload = anthropic_to_openai_payload(parsed.model_dump())
    _rec["oai_model"] = oai_payload.get("model")
    # Preserve original requested model casing for outward-facing responses
    display_model = parsed.model

    # Handle DeepSeek THINKING suffix: strip from upstream model and enable header flag
    thinking_enabled = bool(_orig_en)
    _m, _en = _strip_thinking_suffix(oai_payload.get("model"))
    if _en:
        oai_payload["model"] = _m  # type: ignore
        thinking_enabled = True
    think_trace["payload_suffix"] = bool(_en)
    # Also enable if caller provides explicit thinking/reasoning flags in the request body
    try:
        raw_body = body or {}
        if isinstance(raw_body.get("thinking"), dict):
            t = raw_body["thinking"]
            if (t.get("type") == "enabled") or (int(t.get("budget_tokens", 0)) > 0):
                thinking_enabled = True
                think_trace["body_thinking"] = True
        if isinstance(raw_body.get("reasoning"), dict):
            r = raw_body["reasoning"]
            if r.get("enabled") or int(r.get("max_tokens", 0)) > 0:
                thinking_enabled = True
                think_trace["body_reasoning"] = True
    except Exception:
        ...
    # Respect inbound X-Enable-Thinking header if provided
    if (x_enable_thinking or "").lower() in ("1", "true", "yes", "on", "enable", "enabled"):
        thinking_enabled = True
        think_trace["inbound_header"] = True
    think_trace["final_enabled"] = bool(thinking_enabled)

    # Optional verbose trace for thinking detection
    if settings.debug and os.environ.get("DEBUG_PROXY_VERBOSE", "").lower() in ("1", "true", "yes"):
        try:
            print("[proxy] thinking-detect:", json.dumps(think_trace, ensure_ascii=False))
        except Exception:
            ...

    # Choose streaming or not based on request
    is_stream = bool(oai_payload.get("stream"))
    chutes_url = f"{settings.chutes_base_url.rstrip('/')}/v1/chat/completions"

    if not is_stream:
        # Auto-fix model case if configured (pre-flight)
        if settings.auto_fix_model_case and oai_payload.get("model"):
            try:
                fixed = await _resolve_case_variant(oai_payload["model"], _auth_headers(x_api_key, authorization))
                if not fixed:
                    fixed = _guess_case_for_known_models(oai_payload["model"])  # e.g., moonshotai/Kimi-K2-*
                if fixed and fixed != oai_payload["model"]:
                    if settings.debug:
                        print(f"[proxy] non-stream: auto-fix model case '{oai_payload['model']}' -> '{fixed}'")
                    oai_payload["model"] = fixed
            except Exception:
                ...
        if settings.debug and os.environ.get("DEBUG_PROXY_VERBOSE", "").lower() in ("1", "true", "yes"):
            try:
                print("[proxy] upstream payload (non-stream):", json.dumps({"url": chutes_url, **{k: v for k, v in oai_payload.items() if k != "messages"}, "model": oai_payload.get("model")}))
            except Exception:
                pass
        # Record resolved model (for /_debug/last), without logging sensitive payloads
        _rec["resolved_model"] = oai_payload.get("model")
        client = _get_httpx_client()
        try:
            headers0 = _auth_headers(x_api_key, authorization)
            if thinking_enabled:
                headers0["X-Enable-Thinking"] = "true"
            if settings.debug:
                try:
                    redacted = {
                        k: ("<redacted>" if k.lower() in ("authorization", "x-api-key") else v)
                        for k, v in headers0.items()
                    }
                    print(
                        "[proxy] upstream request (non-stream):",
                        json.dumps({"url": chutes_url, "headers": redacted}, ensure_ascii=False),
                    )
                except Exception:
                    ...
            resp = await client.post(
                chutes_url,
                json=oai_payload,
                headers=headers0,
                timeout=httpx.Timeout(120.0),
            )
        except Exception as e:
            raise HTTPException(status_code=502, detail=f"Upstream error: {e}")

        if resp.status_code >= 400:
            # Limited backoff/retry on 429 if configured
            if resp.status_code == 429 and settings.backoff_on_429 and settings.max_retry_on_429 > 0:
                try:
                    ra = float(resp.headers.get("Retry-After", "0"))
                except Exception:
                    ra = 0.0
                if 0.0 < ra <= settings.max_retry_after_seconds:
                    await asyncio.sleep(ra)
                    try:
                        resp = await client.post(
                            chutes_url,
                            json=oai_payload,
                            headers=headers0,
                            timeout=httpx.Timeout(120.0),
                        )
                    except Exception:
                        ...

            # Map upstream errors to Anthropic-like error envelope (with targeted 404 retry on model casing)
            try:
                data = resp.json()
                message = data.get("error", {}).get("message") or data.get("message") or json.dumps(data)
            except Exception:
                message = resp.text

            # If the error suggests a missing model, attempt one-shot case/alias correction retry
            should_retry_case = False
            try:
                msg_l = (message or "").lower()
                if resp.status_code == 404 and ("model not found" in msg_l or "model" in msg_l and "not found" in msg_l):
                    should_retry_case = True
            except Exception:
                should_retry_case = False

            if should_retry_case:
                try:
                    # Reuse existing headers with potential THINKING flag
                    original_model = str(oai_payload.get("model") or "")
                    # First, try discovery-based resolution
                    alt = await _resolve_case_variant(original_model, headers0)
                    # If still none, try heuristic for known providers (e.g., Moonshot Kimi)
                    if not alt:
                        alt = _guess_case_for_known_models(original_model)
                    if alt and alt != original_model:
                        if settings.debug:
                            try:
                                print(f"[proxy] non-stream: retry on 404 with case-fixed model '{original_model}' -> '{alt}'")
                            except Exception:
                                ...
                        # Retry once with the corrected id
                        retry_payload = dict(oai_payload)
                        retry_payload["model"] = alt
                        _rec["resolved_model_retry"] = alt
                        try:
                            if settings.debug:
                                try:
                                    redacted = {
                                        k: ("<redacted>" if k.lower() in ("authorization", "x-api-key") else v)
                                        for k, v in headers0.items()
                                    }
                                    print(
                                        "[proxy] upstream request (non-stream, retry):",
                                        json.dumps({"url": chutes_url, "headers": redacted}, ensure_ascii=False),
                                    )
                                except Exception:
                                    ...
                            resp2 = await client.post(chutes_url, json=retry_payload, headers=headers0)
                        except Exception:
                            resp2 = None  # type: ignore
                        if resp2 is not None and resp2.status_code < 400:
                            data2 = resp2.json()
                            _rec["phase"] = "non_stream_ok_after_404_retry"
                            _RECENT.append(_rec)
                            return JSONResponse(content=openai_to_anthropic_response(
                                data2,
                                requested_model=display_model,
                                tools=retry_payload.get("tools"),
                                tool_call_parser=choose_tool_call_parser(retry_payload.get("model")),
                            ))
                except Exception:
                    ...

            _rec.update({"phase": "non_stream_error", "upstream_status": resp.status_code, "message": message})
            _RECENT.append(_rec)
            return JSONResponse(status_code=resp.status_code, content=_anthropic_error_for_status(resp.status_code, message))

        data = resp.json()
        _rec["phase"] = "non_stream_ok"
        _RECENT.append(_rec)
        return JSONResponse(content=openai_to_anthropic_response(
            data,
            # Always echo original request model casing outward
            requested_model=display_model,
            tools=oai_payload.get("tools"),
            tool_call_parser=choose_tool_call_parser(oai_payload.get("model")),
        ))

    # Streaming path
    async def event_stream(request: Request) -> AsyncIterator[bytes]:
        # Helper to format SSE (with optional debug printing)
        debug_sse = os.environ.get("DEBUG_SSE", "").lower() in ("1", "true", "yes", "on")
        req_id = uuid.uuid4().hex[:8]
        current_msg_id: Optional[str] = None
        def sse(event: str, data: Dict[str, Any]) -> bytes:
            nonlocal current_msg_id
            try:
                if debug_sse:
                    # Add lightweight correlation for concurrent streams
                    mid = None
                    if event == "message_start":
                        mid = (data.get("message") or {}).get("id")
                    elif event in ("message_delta", "message_stop"):
                        mid = current_msg_id
                    else:
                        mid = current_msg_id
                    print(f"[sse][{req_id}]{'['+str(mid)+']' if mid else ''}", event, json.dumps(data, ensure_ascii=False))
                if event == "message_start":
                    try:
                        current_msg_id = (data.get("message") or {}).get("id")
                    except Exception:
                        current_msg_id = None
            except Exception:
                ...
            return f"event: {event}\n".encode() + f"data: {json.dumps(data, ensure_ascii=False)}\n\n".encode()

        # Open connection to Chutes
        model_to_use = oai_payload.get("model")
        headers0 = _auth_headers(x_api_key, authorization)
        if thinking_enabled:
            headers0["X-Enable-Thinking"] = "true"
        # Map to canonical casing via cached lowercase map (no network RTT)
        if model_to_use:
            try:
                resolved_model = await _resolve_case_variant(model_to_use, headers0)
                if resolved_model and resolved_model != model_to_use:
                    if settings.debug:
                        print(f"[proxy] stream: resolve model '{model_to_use}' -> '{resolved_model}' (cache)")
                    model_to_use = resolved_model
            except Exception:
                ...
        stream_payload = dict(oai_payload)
        # Ensure upstream uses provider's canonical model id; keep outward display via display_model
        if model_to_use:
            stream_payload["model"] = model_to_use
        else:
            # If case-map couldn't resolve but auto-fix is enabled, try minimal fallback cleanup:
            # strip vendor prefix and keep lowercase; upstream may still recognize the suffix-only id.
            try:
                raw = str(stream_payload.get("model") or "")
                if raw:
                    stream_payload["model"] = raw.split("/", 1)[-1]
            except Exception:
                ...
        _rec["resolved_model"] = model_to_use
        if settings.debug and os.environ.get("DEBUG_PROXY_VERBOSE", "").lower() in ("1", "true", "yes"):
            try:
                print("[proxy] upstream payload (stream):", json.dumps({"url": chutes_url, **{k: v for k, v in stream_payload.items() if k != "messages"}, "model": model_to_use}))
            except Exception:
                pass
        client = _get_httpx_client()

        def _open_stream(payload_model: str):
            # Log upstream URL + headers (redacted) for debugging
            _headers_stream = {**headers0, "Accept": "text/event-stream"}
            if settings.debug:
                try:
                    redacted = {
                        k: ("<redacted>" if k.lower() in ("authorization", "x-api-key") else v)
                        for k, v in _headers_stream.items()
                    }
                    print(
                        "[proxy] upstream request (stream):",
                        json.dumps({"url": chutes_url, "headers": redacted}, ensure_ascii=False),
                    )
                except Exception:
                    ...
            return client.stream(
                "POST",
                chutes_url,
                json={**stream_payload, "model": payload_model},
                headers=_headers_stream,
                timeout=None,
            )

        try:
            upstream_cm = _open_stream(model_to_use)
            print(f"[proxy] Opening upstream stream with model: {model_to_use}", file=sys.stderr)
            async with upstream_cm as upstream:
                print(f"[proxy] Upstream stream opened, status: {upstream.status_code}", file=sys.stderr)
                _rec["phase"] = f"stream_status_{upstream.status_code}"
                # If upstream returns an error status, emit an SSE error event and stop
                if upstream.status_code >= 400:
                    try:
                        body_bytes = await upstream.aread()
                        body_text = body_bytes.decode("utf-8", errors="ignore") if body_bytes is not None else ""
                        try:
                            j = json.loads(body_text) if body_text else {}
                            message = (j.get("error") or {}).get("message") or j.get("message") or (json.dumps(j, ensure_ascii=False) if j else "")
                        except Exception:
                            message = body_text
                    except Exception:
                        message = f"Upstream returned HTTP {upstream.status_code} without body"
                    # If it's a 404/400 model not found, try a single retry with case-heuristic and/or suffix-only id
                    if upstream.status_code in (400, 404):
                        # Retry path only once
                        try:
                            retry_model = None
                            # 1) Try canonical case via resolver again (could be a stale cache) 
                            retry_model = await _resolve_case_variant(stream_payload.get("model"), headers0) or None
                            if not retry_model:
                                # 2) Try suffix-only id without vendor prefix
                                raw = str(stream_payload.get("model") or "")
                                retry_model = raw.split("/", 1)[-1] if raw else None
                            if retry_model and retry_model != stream_payload.get("model"):
                                async with _open_stream(retry_model) as upstream2:
                                    _rec["phase"] = f"stream_retry_status_{upstream2.status_code}"
                                    if upstream2.status_code < 400:
                                        # Switch upstream handle transparently
                                        upstream = upstream2
                                        model_to_use = retry_model
                                        stream_payload["model"] = retry_model
                                        # Fall through into normal SSE handling without re-emitting error
                                        ct2 = upstream.headers.get("content-type") or upstream.headers.get("Content-Type") or ""
                                        if "text/event-stream" not in str(ct2).lower():
                                            # For non-SSE retry, read once and map similar to main path below
                                            try:
                                                raw2 = await upstream.aread()
                                                txt2 = raw2.decode("utf-8", errors="ignore") if raw2 is not None else ""
                                                data_obj2 = json.loads(txt2) if txt2 else {}
                                            except Exception:
                                                data_obj2 = {}
                                            if not data_obj2:
                                                yield sse("error", {"type": "error", "error": {"type": "api_error", "message": "Upstream returned no SSE and empty body"}})
                                                yield sse("message_stop", {"type": "message_stop"})
                                                _rec["phase"] = "stream_retry_fallback_empty_body"
                                                _RECENT.append(_rec)
                                                return
                                            anth2 = openai_to_anthropic_response(
                                                data_obj2,
                                                requested_model=display_model,
                                                tools=stream_payload.get("tools"),
                                                tool_call_parser=choose_tool_call_parser(model_to_use),
                                            )
                                            ms2 = ensure_message_start()
                                            if ms2:
                                                yield ms2
                                            for blk in (anth2.get("content") or []):
                                                btype = blk.get("type")
                                                if btype == "thinking":
                                                    cb_index = next_block_index
                                                    next_block_index += 1
                                                    yield sse("content_block_start", {"type": "content_block_start", "index": cb_index, "content_block": {"type": "thinking", "thinking": ""}})
                                                    yield sse("content_block_delta", {"type": "content_block_delta", "index": cb_index, "delta": {"type": "thinking_delta", "thinking": blk.get("thinking", "")}})
                                                    yield sse("content_block_stop", {"type": "content_block_stop", "index": cb_index})
                                                elif btype == "text":
                                                    ts2 = ensure_text_block()
                                                    if ts2:
                                                        yield ts2
                                                    yield sse("content_block_delta", {"type": "content_block_delta", "index": text_block_index, "delta": {"type": "text_delta", "text": blk.get("text", "")}})
                                                elif btype == "tool_use":
                                                    cb_index = next_block_index
                                                    next_block_index += 1
                                                    tid = blk.get("id") or f"call_{uuid.uuid4().hex}"
                                                    tname = blk.get("name") or ""
                                                    tinput = blk.get("input") or {}
                                                    yield sse("content_block_start", {"type": "content_block_start", "index": cb_index, "content_block": {"type": "tool_use", "id": tid, "name": tname, "input": {}}})
                                                    try:
                                                        args_str = json.dumps(tinput, ensure_ascii=False)
                                                    except Exception:
                                                        args_str = "{}"
                                                    yield sse("content_block_delta", {"type": "content_block_delta", "index": cb_index, "delta": {"type": "input_json_delta", "partial_json": args_str}})
                                                    yield sse("content_block_stop", {"type": "content_block_stop", "index": cb_index})
                                                    emitted_tool_use = True
                                            if text_block_index is not None:
                                                yield sse("content_block_stop", {"type": "content_block_stop", "index": text_block_index})
                                                text_block_index = None
                                            usage_obj2 = anth2.get("usage") or {}
                                            stop_reason2 = anth2.get("stop_reason") or ("tool_use" if emitted_tool_use else "end_turn")
                                            yield sse("message_delta", {"type": "message_delta", "delta": {"stop_reason": stop_reason2, "stop_sequence": None}, "usage": {"input_tokens": int(usage_obj2.get("input_tokens") or 0), "output_tokens": int(usage_obj2.get("output_tokens") or 0)}})
                                            yield sse("message_stop", {"type": "message_stop"})
                                            _rec["phase"] = "stream_retry_fallback_non_sse"
                                            _RECENT.append(_rec)
                                            return
                                        # if SSE, continue into normal streaming loop
                                    # If retry still fails, fall through to emit original error below
                        except Exception:
                            ...
                    try:
                        err_env = _anthropic_error_for_status(upstream.status_code, message or "")
                        etype = (err_env.get("error") or {}).get("type") or "api_error"
                    except Exception:
                        etype = "api_error"
                    # Emit a well-formed message envelope even on error to avoid client-side abrupt end
                    ms0 = ensure_message_start()
                    if ms0:
                        yield ms0
                    yield sse("error", {"type": "error", "error": {"type": etype, "message": message or "Upstream error"}})
                    yield sse("message_stop", {"type": "message_stop"})
                    _rec["phase"] = "stream_error_status"
                    _rec["upstream_status"] = upstream.status_code
                    _rec["message"] = message
                    _RECENT.append(_rec)
                    return

                # If upstream did not return SSE content-type, try to fallback by mapping the JSON body to SSE
                ct = upstream.headers.get("content-type") or upstream.headers.get("Content-Type") or ""
                if "text/event-stream" not in str(ct).lower():
                    try:
                        raw = await upstream.aread()
                        txt = raw.decode("utf-8", errors="ignore") if raw is not None else ""
                        data_obj = json.loads(txt) if txt else {}
                    except Exception:
                        data_obj = {}
                    if not data_obj:
                        ms1 = ensure_message_start()
                        if ms1:
                            yield ms1
                        yield sse("error", {"type": "error", "error": {"type": "api_error", "message": "Upstream returned no SSE and empty body"}})
                        yield sse("message_stop", {"type": "message_stop"})
                        _rec["phase"] = "stream_fallback_empty_body"
                        _RECENT.append(_rec)
                        return
                    anth = openai_to_anthropic_response(
                        data_obj,
                        requested_model=display_model,
                        tools=stream_payload.get("tools"),
                        tool_call_parser=choose_tool_call_parser(model_to_use),
                    )
                    ms = ensure_message_start()
                    if ms:
                        yield ms
                    # Emit content blocks reconstructed from non-stream response
                    for blk in (anth.get("content") or []):
                        btype = blk.get("type")
                        if btype == "thinking":
                            cb_index = next_block_index
                            next_block_index += 1
                            yield sse(
                                "content_block_start",
                                {
                                    "type": "content_block_start",
                                    "index": cb_index,
                                    "content_block": {"type": "thinking", "thinking": ""},
                                },
                            )
                            yield sse(
                                "content_block_delta",
                                {
                                    "type": "content_block_delta",
                                    "index": cb_index,
                                    "delta": {"type": "thinking_delta", "thinking": blk.get("thinking", "")},
                                },
                            )
                            yield sse("content_block_stop", {"type": "content_block_stop", "index": cb_index})
                        elif btype == "text":
                            ts = ensure_text_block()
                            if ts:
                                yield ts
                            yield sse(
                                "content_block_delta",
                                {
                                    "type": "content_block_delta",
                                    "index": text_block_index,
                                    "delta": {"type": "text_delta", "text": blk.get("text", "")},
                                },
                            )
                        elif btype == "tool_use":
                            cb_index = next_block_index
                            next_block_index += 1
                            tid = blk.get("id") or f"call_{uuid.uuid4().hex}"
                            tname = blk.get("name") or ""
                            tinput = blk.get("input") or {}
                            yield sse(
                                "content_block_start",
                                {
                                    "type": "content_block_start",
                                    "index": cb_index,
                                    "content_block": {"type": "tool_use", "id": tid, "name": tname, "input": {}},
                                },
                            )
                            try:
                                args_str = json.dumps(tinput, ensure_ascii=False)
                            except Exception:
                                args_str = "{}"
                            yield sse(
                                "content_block_delta",
                                {
                                    "type": "content_block_delta",
                                    "index": cb_index,
                                    "delta": {"type": "input_json_delta", "partial_json": args_str},
                                },
                            )
                            yield sse("content_block_stop", {"type": "content_block_stop", "index": cb_index})
                            emitted_tool_use = True
                    if text_block_index is not None:
                        yield sse("content_block_stop", {"type": "content_block_stop", "index": text_block_index})
                        text_block_index = None
                    usage_obj = anth.get("usage") or {}
                    stop_reason = anth.get("stop_reason") or ("tool_use" if emitted_tool_use else "end_turn")
                    yield sse(
                        "message_delta",
                        {
                            "type": "message_delta",
                            "delta": {"stop_reason": stop_reason, "stop_sequence": None},
                            "usage": {
                                "input_tokens": int(usage_obj.get("input_tokens") or 0),
                                "output_tokens": int(usage_obj.get("output_tokens") or 0),
                            },
                        },
                    )
                    yield sse("message_stop", {"type": "message_stop"})
                    _rec["phase"] = "stream_fallback_non_sse"
                    _RECENT.append(_rec)
                    return

                # Guard: if no chunks are received at all, emit an error event instead of silently ending
                no_chunks = True
                print("[proxy] Starting stream processing", file=sys.stderr)
                # Anthropic stream scaffold
                final_text: list[str] = []
                sent_start = False
                model_name = None
                usage_input = 0
                usage_output = 0
                emitted_tool_use = False

                # Content block management
                next_block_index = 0
                text_block_index: int | None = None
                tool_states: Dict[int, Dict[str, Any]] = {}

                def ensure_message_start():
                    nonlocal sent_start
                    if not sent_start:
                        msg = {
                            "id": f"msg_{uuid.uuid4().hex}",
                            "type": "message",
                            "role": "assistant",
                            # Outward-facing: always echo the originally requested casing
                            "model": display_model or model_name or stream_payload.get("model", "unknown-model"),
                            # Start with empty content to align with Anthropic stream semantics
                            "content": [],
                            # Provide a usage object upfront to satisfy clients that read it early
                            "usage": {"input_tokens": 0, "output_tokens": 0},
                            "stop_reason": None,
                            "stop_sequence": None,
                        }
                        sent_start = True
                        return sse("message_start", {"type": "message_start", "message": msg})
                    return None

                def ensure_text_block():
                    nonlocal text_block_index, next_block_index
                    if text_block_index is None:
                        text_block_index = next_block_index
                        next_block_index += 1
                        return sse(
                            "content_block_start",
                            {
                                "type": "content_block_start",
                                "index": text_block_index,
                                "content_block": {"type": "text", "text": ""},
                            },
                        )
                    return None

                # Prepare sglang FunctionCallParser if tools exist and enabled
                parser = None
                parser_name = choose_tool_call_parser(model_to_use)
                if (stream_payload.get("tools")) and settings.enable_stream_tool_parser:
                    try:
                        from sglang.srt.function_call.function_call_parser import FunctionCallParser as _FCP

                        class _Fn:
                            def __init__(self, name: str, parameters: Dict[str, Any], strict: bool = False):
                                self.name = name
                                self.parameters = parameters
                                self.strict = strict

                        class _Tool:
                            def __init__(self, fn: _Fn):
                                self.function = fn

                        _tool_objs = []
                        for t in stream_payload.get("tools") or []:
                            fn = (t or {}).get("function") or {}
                            name = fn.get("name") or ""
                            params = fn.get("parameters") or {}
                            strict = bool(fn.get("strict", False))
                            if name:
                                _tool_objs.append(_Tool(_Fn(name, params, strict)))
                        if _tool_objs:
                            parser = _FCP(_tool_objs, parser_name)
                    except Exception:
                        parser = None

                # Rely solely on sglang parser when available

                # Buffer for Cloud Code MCP tool-call markup across stream chunks
                mcp_buf = ""

                # Tool names come only from the current request's declared tools
                try:
                    known_tools: list[str] = [t.name for t in (parsed.tools or []) if getattr(t, "name", None)]
                except Exception:
                    known_tools = []
                if settings.debug and os.environ.get("DEBUG_PROXY_VERBOSE", "").lower() in ("1", "true", "yes"):
                    try:
                        print("[proxy] declared tools:", json.dumps(known_tools, ensure_ascii=False))
                    except Exception:
                        ...

                # Normalize a raw tool name to the best matching declared tool
                def _normalize_tool_name(raw_name: Optional[str]) -> str:
                    try:
                        name_in = (raw_name or "").strip()
                        if not name_in:
                            return ""
                        # Common synonyms fallback
                        synonyms = {
                            "grep": "Grep",
                            "glob": "Glob",
                            "globfiles": "Glob",
                            "todowrite": "TodoWrite",
                            "todo": "TodoWrite",
                            "bash": "Bash",
                            "shell": "Bash",
                            "run": "Bash",
                            "runcmd": "Bash",
                            "runcmds": "Bash",
                            "runcommand": "Bash",
                            "terminal": "Bash",
                            "sh": "Bash",
                            # Additional common aliases observed in MCP tool markup
                            "terminal_execute": "Bash",
                            "terminal-execute": "Bash",
                            "terminalexec": "Bash",
                            "terminal_run": "Bash",
                            "execute_terminal": "Bash",
                            # Additional synonyms for better compatibility
                            "run_terminal_cmd": "Bash",
                            "glob_file_search": "Glob",
                            "todo_write": "TodoWrite",
                        }
                        name_in = synonyms.get(name_in) or synonyms.get(name_in.lower()) or name_in
                        # 1) exact match against declared
                        if name_in in known_tools:
                            return name_in
                        # 2) case-insensitive match against declared
                        low = name_in.lower()
                        for kt in known_tools:
                            try:
                                if str(kt).lower() == low:
                                    return kt
                            except Exception:
                                ...
                        # 3) env-configured mapping
                        try:
                            mapped = settings.map_tool_name(model_to_use, name_in)
                        except Exception:
                            mapped = name_in
                        mapped = mapped or name_in
                        if isinstance(mapped, str):
                            mlow = mapped.strip().lower()
                            for kt in known_tools:
                                try:
                                    if str(kt).lower() == mlow:
                                        return kt
                                except Exception:
                                    ...
                        # 4) fuzzy to declared: compare alnum-only lowercase forms and containment
                        import re as _re
                        def _norm(s: str) -> str:
                            return _re.sub(r"[^a-z0-9]", "", s.lower())
                        n_in = _norm(name_in)
                        best = None
                        best_score = -1
                        for kt in known_tools:
                            try:
                                n_kt = _norm(str(kt))
                                score = 0
                                if n_kt == n_in:
                                    score = 100
                                elif n_in and n_in in n_kt:
                                    score = 80 - max(0, len(n_kt) - len(n_in))
                                elif n_kt and n_kt in n_in:
                                    score = 60 - max(0, len(n_in) - len(n_kt))
                                if score > best_score:
                                    best_score = score
                                    best = kt
                            except Exception:
                                ...
                        if best is not None and best_score >= 60:
                            return best
                        # 5) snake_case fallback for general normalization
                        try:
                            s1 = _re.sub(r"(.)([A-Z][a-z]+)", r"\\1_\\2", name_in)
                            s2 = _re.sub(r"([a-z0-9])([A-Z])", r"\\1_\\2", s1)
                            out = s2.replace(" ", "_").replace("-", "_").lower()
                        except Exception:
                            out = name_in
                        # prefer any declared that equals this fallback (case-insensitive)
                        low2 = (out or "").lower()
                        for kt in known_tools:
                            try:
                                if str(kt).lower() == low2:
                                    return kt
                            except Exception:
                                ...
                        return out
                    except Exception:
                        return raw_name or ""

                # Normalize tool input args to expected schema for known tools
                def _normalize_tool_input(tool_name: Optional[str], args_obj: Any) -> Any:
                    try:
                        if not isinstance(args_obj, dict):
                            return args_obj
                        name = (_normalize_tool_name(tool_name) or "").lower()
                        args = dict(args_obj)
                        if name == "grep":
                            # field aliases
                            if "query" in args and "pattern" not in args:
                                args["pattern"] = args.pop("query")
                            if "regex" in args and "pattern" not in args:
                                args["pattern"] = args.pop("regex")
                            # path aliases
                            for k in ("directory", "dir", "root", "file", "files"):
                                if k in args and "path" not in args and isinstance(args.get(k), str):
                                    args["path"] = args.pop(k)
                                    break
                            # glob aliases
                            for k in ("glob_pattern", "globpattern", "globPath"):
                                if k in args and "glob" not in args:
                                    args["glob"] = args.pop(k)
                                    break
                            # boolean/flags
                            if args.pop("-n", None) is not None:
                                # tool doesn't support -n; ignore
                                pass
                            if "case_insensitive" in args and "-i" not in args:
                                try:
                                    args["-i"] = bool(args.pop("case_insensitive"))
                                except Exception:
                                    args.pop("case_insensitive", None)
                            # output mode helpers
                            if isinstance(args.get("count"), bool) and args.get("count"):
                                args["output_mode"] = "count"
                                args.pop("count", None)
                            if isinstance(args.get("files_with_matches"), bool) and args.get("files_with_matches"):
                                args["output_mode"] = "files_with_matches"
                                args.pop("files_with_matches", None)
                            # context mapping
                            for s, d in (("before", "-B"), ("after", "-A"), ("context", "-C")):
                                if s in args and d not in args:
                                    try:
                                        args[d] = int(args.pop(s))
                                    except Exception:
                                        args.pop(s, None)
                            # limits & typing
                            if "limit" in args and "head_limit" not in args:
                                try:
                                    args["head_limit"] = int(args.pop("limit"))
                                except Exception:
                                    args.pop("limit", None)
                            if "multiline" in args:
                                try:
                                    args["multiline"] = bool(args["multiline"])
                                except Exception:
                                    args.pop("multiline", None)
                            return args
                        if name == "glob_file_search":
                            if "pattern" in args and "glob_pattern" not in args:
                                args["glob_pattern"] = args.pop("pattern")
                            for k in ("dir", "directory", "root", "path"):
                                if k in args and "target_directory" not in args and isinstance(args.get(k), str):
                                    args["target_directory"] = args.pop(k)
                                    break
                            return args
                        if name == "run_terminal_cmd":
                            # default is_background to False
                            if "is_background" not in args:
                                bg_alias = args.pop("background", None)
                                try:
                                    args["is_background"] = bool(bg_alias) if bg_alias is not None else False
                                except Exception:
                                    args["is_background"] = False
                            # cmd aliases
                            if "command" not in args and "cmd" in args:
                                args["command"] = args.pop("cmd")
                            # join args array into command
                            if "command" not in args and isinstance(args.get("args"), list):
                                try:
                                    args["command"] = " ".join(map(str, args.pop("args") or []))
                                except Exception:
                                    args.pop("args", None)
                            # quote common glob patterns for find -name
                            try:
                                import re as _re
                                cmd = str(args.get("command") or "")
                                def _quote_name_pattern(m: "_re.Match[str]") -> str:
                                    token = m.group(1)
                                    if not token:
                                        return m.group(0)
                                    if ("*" in token or "?" in token or "[" in token) and not (token.startswith("\"") or token.startswith("'")):
                                        return "-name '" + token + "'"
                                    return m.group(0)
                                cmd2 = _re.sub(r"-name\s+([^\s'\"][^\s]*)", _quote_name_pattern, cmd)
                                if cmd2 != cmd:
                                    args["command"] = cmd2
                            except Exception:
                                ...
                            return args
                        return args_obj
                    except Exception:
                        return args_obj

                # Track a dedicated thinking block for reasoning deltas
                thinking_block_index: Optional[int] = None
                thinking_open = False
                # Buffer for MCP tool markup that appears inside thinking content
                think_mcp_buf = ""
                # Buffer for any normal text that arrives in the same chunk as thinking
                pending_text_after_thinking = ""

                # Hysteresis to avoid rapid flip from thinking->text within the very next chunk
                try:
                    hysteresis = int(os.environ.get("REASONING_HYSTERESIS", "2"))
                except Exception:
                    hysteresis = 2
                no_reasoning_streak = 0

                async for line in upstream.aiter_lines():
                        # Check if client has disconnected
                        if await request.is_disconnected():
                            print("[proxy] Client disconnected during streaming", file=sys.stderr)
                            # Close any open blocks before ending
                            if text_block_index is not None:
                                yield sse("content_block_stop", {"type": "content_block_stop", "index": text_block_index})
                            if thinking_open and thinking_block_index is not None:
                                yield sse("content_block_stop", {"type": "content_block_stop", "index": thinking_block_index})
                            for st in list(tool_states.values()):
                                if st.get("open"):
                                    yield sse("content_block_stop", {"type": "content_block_stop", "index": st["cb_index"]})
                            # Emit final message delta and stop
                            yield sse(
                                "message_delta",
                                {
                                    "type": "message_delta",
                                    "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                                    "usage": {"input_tokens": usage_input, "output_tokens": usage_output},
                                },
                            )
                            yield sse("message_stop", {"type": "message_stop"})
                            _rec["phase"] = "stream_client_disconnected"
                            _RECENT.append(_rec)
                            print("[proxy] Stream ended due to client disconnection", file=sys.stderr)
                            return

                        if not line:
                            print("[proxy] Received empty line", file=sys.stderr)
                            continue
                        if not line.startswith("data:"):
                            print(f"[proxy] Received non-data line: {line[:100]}", file=sys.stderr)
                            continue
                        no_chunks = False
                        payload_str = line[5:].strip()
                        print(f"[proxy] Received stream line: {payload_str[:100]}...", file=sys.stderr)
                        if payload_str == "[DONE]":
                            # Close any open blocks
                            if text_block_index is not None:
                                yield sse("content_block_stop", {"type": "content_block_stop", "index": text_block_index})
                                text_block_index = None
                            if thinking_open and thinking_block_index is not None:
                                yield sse("content_block_stop", {"type": "content_block_stop", "index": thinking_block_index})
                                thinking_open = False
                                thinking_block_index = None
                            for st in list(tool_states.values()):
                                if st.get("open"):
                                    yield sse("content_block_stop", {"type": "content_block_stop", "index": st["cb_index"]})
                                    st["open"] = False
                            # Final message_delta and stop
                            final_stop = "tool_use" if emitted_tool_use else "end_turn"
                            yield sse(
                                "message_delta",
                                {
                                    "type": "message_delta",
                                    "delta": {"stop_reason": final_stop, "stop_sequence": None},
                                    "usage": {"input_tokens": usage_input, "output_tokens": usage_output},
                                },
                            )
                            yield sse("message_stop", {"type": "message_stop"})
                            _rec["phase"] = "stream_ok"
                            _RECENT.append(_rec)
                            break
                        try:
                            chunk = json.loads(payload_str)
                        except Exception:
                            continue

                        # Extract model
                        if not model_name:
                            # Prefer requested model for case preservation
                            model_name = stream_payload.get("model") or chunk.get("model")

                        # Accumulate text deltas
                        choices = chunk.get("choices") or []
                        if choices:
                            delta = (choices[0].get("delta") or {})

                            # Handle reasoning/thinking stream first
                            # Support either OpenAI reasoning models (delta.reasoning_content)
                            # or upstream-transformed shape (delta.thinking.{content|signature})
                            reasoning_piece = delta.get("reasoning_content") or None
                            thinking_obj = delta.get("thinking") or {}
                            thinking_signature = None
                            if isinstance(thinking_obj, dict):
                                thinking_signature = thinking_obj.get("signature")
                                if not reasoning_piece and thinking_obj.get("content"):
                                    reasoning_piece = thinking_obj.get("content")

                            if reasoning_piece:
                                no_reasoning_streak = 0
                                if not sent_start:
                                    ms = ensure_message_start()
                                    if ms:
                                        yield ms
                                # Normalize potential fullwidth markers used by some models
                                try:
                                    norm_reason = (
                                        str(reasoning_piece)
                                        .replace("｜", "|")
                                        .replace("▁", "_")
                                    )
                                except Exception:
                                    norm_reason = str(reasoning_piece)

                                # Accumulate thinking raw text into buffer
                                think_mcp_buf += norm_reason

                                # Helper: drain unmarked calls like "ToolName{...json...}" from buffer or plain text.
                                # Returns (plain_text, calls_for_declared_tools, leftover, moved_unknown)
                                def _drain_unmarked_calls(buf: str, tool_names: list[str], hold_tail_for_declared: bool = True):
                                    import re as _re
                                    import json as _json
                                    # If we know no tool names, still detect generic pattern to strip from thinking
                                    # Generic name pattern keeps it provider-agnostic, no hardcoding of names
                                    declared = set([n for n in (tool_names or []) if isinstance(n, str) and n])
                                    pat = _re.compile(r"\b([A-Za-z][A-Za-z0-9_\-\.]{0,80})\s*\{", _re.DOTALL)
                                    out_parts: list[str] = []
                                    calls: list[dict] = []
                                    moved_unknown: list[str] = []
                                    pos = 0
                                    n = len(buf)
                                    while pos < n:
                                        m = pat.search(buf, pos)
                                        if not m:
                                            out_parts.append(buf[pos:])
                                            break
                                        start = m.start()
                                        name = (m.group(1) or "").strip()
                                        # Emit preceding text
                                        out_parts.append(buf[pos:start])
                                        # Find balanced JSON object starting at the "{" right after the match
                                        brace_start = m.end() - 1  # points at '{'
                                        i = brace_start
                                        depth = 0
                                        in_str = False
                                        esc = False
                                        end = -1
                                        while i < n:
                                            ch = buf[i]
                                            if esc:
                                                esc = False
                                            elif ch == "\\":
                                                esc = True
                                            elif ch == '"':
                                                in_str = not in_str
                                            elif not in_str:
                                                if ch == "{":
                                                    depth += 1
                                                elif ch == "}":
                                                    depth -= 1
                                                    if depth == 0:
                                                        end = i
                                                        break
                                            i += 1
                                        # If not complete, keep the whole match for next chunk
                                        if end == -1:
                                            # Put back the unmatched segment
                                            leftover = buf[start:]
                                            return "".join(out_parts), calls, leftover, ""
                                        # Try to parse JSON slice
                                        json_slice = buf[brace_start : end + 1]
                                        try:
                                            args = _json.loads(json_slice)
                                        except Exception:
                                            # Not valid JSON yet; keep for next chunk
                                            leftover = buf[start:]
                                            return "".join(out_parts), calls, leftover, ""
                                        # It is a valid call: record and advance
                                        if name in declared:
                                            calls.append({
                                                "type": "tool_use",
                                                "id": f"call_{uuid.uuid4().hex}",
                                                "name": name,
                                                "input": args if isinstance(args, dict) else {},
                                            })
                                        else:
                                            # Unknown tool name: move out of thinking to normal text later
                                            moved_unknown.append(f"{name}{json_slice}")
                                        pos = end + 1
                                    text_out = "".join(out_parts)
                                    leftover = ""
                                    # Hold back a trailing declared tool name without '{' to avoid leaking (e.g., 'Bash' then next chunk '{...}')
                                    if hold_tail_for_declared and text_out:
                                        tail = text_out[-128:]
                                        # First, prefer declared names when provided
                                        if declared:
                                            for name in sorted(declared, key=len, reverse=True):
                                                if not name:
                                                    continue
                                                if tail.rstrip().endswith(name):
                                                    idx = text_out.rstrip().rfind(name)
                                                    if idx != -1 and (idx == 0 or not text_out[idx-1].isalnum()):
                                                        keep = text_out[:idx]
                                                        hold = text_out[idx:]
                                                        text_out = keep
                                                        leftover = hold + leftover
                                                        break
                                        else:
                                            # Generic conservative hold: if the suffix looks like a single identifier token
                                            # (no punctuation, <= 40 chars), keep it in leftover to wait for a possible '{' next chunk.
                                            # This avoids leaking sequences like 'Read' or 'Write' into thinking.
                                            import re as _re2
                                            m = _re2.search(r"([A-Za-z][A-Za-z0-9_\-\.]{0,39})\s*$", tail)
                                            if m:
                                                token = m.group(1)
                                                # Avoid common short English words that are unlikely to be tool names
                                                if token.lower() not in {"a", "an", "the", "and", "or", "to", "in", "on", "of", "for"}:
                                                    idx = text_out.rfind(token)
                                                    if idx != -1:
                                                        keep = text_out[:idx]
                                                        hold = text_out[idx:]
                                                        text_out = keep
                                                        leftover = hold + leftover
                                    return text_out, calls, leftover, "".join(moved_unknown)

                                # If this chunk also carries normal content, buffer it for next non-thinking chunk to avoid mixing
                                aux_text = delta.get("content") or ""
                                if aux_text:
                                    try:
                                        aux_text = (
                                            str(aux_text).replace("｜", "|").replace("▁", "_")
                                        )
                                    except Exception:
                                        aux_text = str(aux_text)
                                    pending_text_after_thinking += aux_text

                                def _parse_simple_calls(section: str):
                                    import re as _re
                                    simple_calls: list[dict] = []
                                    try:
                                        # Try multiple patterns to handle different character encodings
                                        patterns = [
                                            r"<\|tool_call_begin\|>(.*?)<\|tool_sep\|>(.*?)<\|tool_call_end\|>",
                                            r"<\|tool_call_begin\|>(.*?)<\|tool_sep\|>(.*?)(?:<\|tool_call_end\|>|\s*$)",
                                            r"<｜tool▁call▁begin｜>(.*?)<｜tool▁sep｜>(.*?)<｜tool▁call▁end｜>",
                                            r"<｜tool▁call▁begin｜>(.*?)<｜tool▁sep｜>(.*?)(?:<｜tool▁call▁end｜>|\s*$)"
                                        ]

                                        for pattern_str in patterns:
                                            try:
                                                pattern = _re.compile(pattern_str, _re.DOTALL)
                                                for m in pattern.finditer(section):
                                                    name = (m.group(1) or '').strip()
                                                    args_raw = (m.group(2) or '').strip()
                                                    try:
                                                        import json as _json
                                                        args = _json.loads(args_raw)
                                                    except Exception:
                                                        args = {}
                                                    simple_calls.append({
                                                        "type": "tool_use",
                                                        "id": f"call_{uuid.uuid4().hex}",
                                                        "name": name,
                                                        "input": args if isinstance(args, dict) else {}
                                                    })
                                                # If we found matches with this pattern, break
                                                if simple_calls:
                                                    break
                                            except Exception:
                                                continue
                                    except Exception as e:
                                        # Log the error for debugging but don't crash
                                        print(f"[proxy] Error parsing simple calls: {e}", file=sys.stderr)
                                        # Try alternative parsing for malformed tool calls
                                        try:
                                            # Handle case where tool calls might be malformed
                                            tools = ["TodoWrite", "Read", "Write", "Edit", "Grep", "Glob", "Bash"]
                                            for tool_name in tools:
                                                if tool_name in section:
                                                    # Try to extract tool calls with a more permissive pattern
                                                    alt_pattern = _re.compile(rf"{tool_name}[^\{{]*(\{{.*?\}})")
                                                    for m in alt_pattern.finditer(section):
                                                        try:
                                                            import json as _json
                                                            args = _json.loads(m.group(1))
                                                            simple_calls.append({
                                                                "type": "tool_use",
                                                                "id": f"call_{uuid.uuid4().hex}",
                                                                "name": tool_name,
                                                                "input": args if isinstance(args, dict) else {}
                                                            })
                                                        except Exception:
                                                            pass
                                        except Exception:
                                            pass
                                    return simple_calls

                                def _is_marker_only(text: str) -> bool:
                                    try:
                                        import re as _re
                                        s = text or ""
                                        # Remove all known tool markers
                                        s = _re.sub(r"<\|tool_(calls_)?(begin|end)\|>", "", s)
                                        s = _re.sub(r"<\|tool_call_(begin|end)\|>", "", s)
                                        s = _re.sub(r"<\|tool_sep\|>", "", s)
                                        # Drop whitespace/newlines
                                        s = s.strip()
                                        return len(s) == 0
                                    except Exception:
                                        return False

                                def _drain_mcp_thinking(buf: str):
                                    BEG1 = "<|tool_calls_section_begin|>"; END1 = "<|tool_calls_section_end|>"
                                    BEG2 = "<|tool_calls_begin|>"; END2 = "<|tool_calls_end|>"
                                    out_text_parts: list[str] = []
                                    calls: list[dict] = []
                                    pos = 0
                                    n = len(buf)
                                    while pos < n:
                                        i1 = buf.find(BEG1, pos); i2 = buf.find(BEG2, pos)
                                        # pick earliest begin marker
                                        candidates = [x for x in [i1, i2] if x != -1]
                                        if not candidates:
                                            out_text_parts.append(buf[pos:])
                                            break
                                        i = min(candidates)
                                        out_text_parts.append(buf[pos:i])
                                        if i == i1:
                                            j = buf.find(END1, i)
                                            if j == -1:
                                                # incomplete
                                                return "".join(out_text_parts), calls, buf[i:]
                                            section = buf[i:j+len(END1)]
                                            norm_text, tcalls = parse_mcp_tool_markup(section)
                                            if norm_text:
                                                out_text_parts.append(norm_text)
                                            if tcalls:
                                                calls.extend(tcalls)
                                            pos = j + len(END1)
                                        else:
                                            j = buf.find(END2, i)
                                            if j == -1:
                                                return "".join(out_text_parts), calls, buf[i:]
                                            section = buf[i:j+len(END2)]
                                            tcalls = _parse_simple_calls(section)
                                            # simple style has no replacement text
                                            if tcalls:
                                                calls.extend(tcalls)
                                            pos = j + len(END2)
                                    return "".join(out_text_parts), calls, ""

                                # Drain markers first against the entire buffer
                                drained1, calls1, left1 = _drain_mcp_thinking(think_mcp_buf)
                                # Then drain unmarked calls against the plain text returned from marker drain
                                drained2, calls2, left2, moved_unknown = _drain_unmarked_calls(drained1, known_tools, hold_tail_for_declared=True)

                                drained_think_text = drained2  # only fully safe plain text
                                think_calls = []
                                if calls1:
                                    think_calls.extend(calls1)
                                if calls2:
                                    think_calls.extend(calls2)
                                # Leftover may be from open tool section or partial unmarked JSON; keep both
                                think_mcp_buf = (left1 or "") + (left2 or "")
                                if moved_unknown:
                                    pending_text_after_thinking += moved_unknown

                                # If there are calls found inside the thinking stream, close thinking block before tool_use
                                if think_calls:
                                    # Emit any accumulated thinking text first (skip if only markers/whitespace)
                                    if drained_think_text and not _is_marker_only(drained_think_text):
                                        if not thinking_open:
                                            cb_index = next_block_index
                                            next_block_index += 1
                                            thinking_block_index = cb_index
                                            thinking_open = True
                                            yield sse(
                                                "content_block_start",
                                                {
                                                    "type": "content_block_start",
                                                    "index": cb_index,
                                                    "content_block": {"type": "thinking", "thinking": ""},
                                                },
                                            )
                                        yield sse(
                                            "content_block_delta",
                                            {
                                                "type": "content_block_delta",
                                                "index": thinking_block_index,
                                                "delta": {"type": "thinking_delta", "thinking": drained_think_text},
                                            },
                                        )
                                    if thinking_open and thinking_block_index is not None:
                                        yield sse(
                                            "content_block_stop",
                                            {"type": "content_block_stop", "index": thinking_block_index},
                                        )
                                        thinking_open = False
                                        thinking_block_index = None

                                    # Emit tool_use blocks for each parsed call
                                for call in think_calls:
                                    cb_index = next_block_index
                                    next_block_index += 1
                                    call_id = call.get("id") or f"call_{uuid.uuid4().hex}"
                                    call_name = _normalize_tool_name(call.get("name")) or "tool"
                                    call_args = call.get("input") or {}
                                    yield sse(
                                        "content_block_start",
                                        {
                                            "type": "content_block_start",
                                            "index": cb_index,
                                            "content_block": {
                                                "type": "tool_use",
                                                "id": call_id,
                                                "name": call_name,
                                                "input": {},
                                            },
                                        },
                                    )
                                    try:
                                        import json as _json
                                        args_str = _json.dumps(call_args, ensure_ascii=False)
                                    except Exception:
                                        args_str = "{}"
                                    yield sse(
                                        "content_block_delta",
                                        {
                                            "type": "content_block_delta",
                                            "index": cb_index,
                                            "delta": {
                                                "type": "input_json_delta",
                                                "partial_json": args_str,
                                            },
                                        },
                                    )
                                    yield sse(
                                        "content_block_stop",
                                        {"type": "content_block_stop", "index": cb_index},
                                    )
                                    emitted_tool_use = True
                                    # After tool_use, we keep thinking closed; it will reopen on next reasoning chunk
                                else:
                                    # No complete tool calls found in this reasoning chunk.
                                    # Only stream the drained plain text (excludes any partial unmarked-call leftovers).
                                    if drained_think_text and not _is_marker_only(drained_think_text):
                                        if not thinking_open:
                                            cb_index = next_block_index
                                            next_block_index += 1
                                            thinking_block_index = cb_index
                                            thinking_open = True
                                            yield sse(
                                                "content_block_start",
                                                {
                                                    "type": "content_block_start",
                                                    "index": cb_index,
                                                    "content_block": {"type": "thinking", "thinking": ""},
                                                },
                                            )
                                        yield sse(
                                            "content_block_delta",
                                            {
                                                "type": "content_block_delta",
                                                "index": thinking_block_index,
                                                "delta": {"type": "thinking_delta", "thinking": drained_think_text},
                                            },
                                        )
                                    # If nothing was drained (likely we're mid-JSON), hold output to avoid leaking JSON
                                # Since we handled thinking (and possibly tool) in this chunk, do not process normal text again
                                continue
                            # Signature ends the thinking block if provided
                            if thinking_signature and thinking_open and thinking_block_index is not None:
                                yield sse(
                                    "content_block_delta",
                                    {
                                        "type": "content_block_delta",
                                        "index": thinking_block_index,
                                        "delta": {"type": "signature_delta", "signature": thinking_signature},
                                    },
                                )
                                yield sse(
                                    "content_block_stop",
                                    {"type": "content_block_stop", "index": thinking_block_index},
                                )
                                thinking_open = False
                                thinking_block_index = None

                            # Text piece
                            piece = delta.get("content") or ""
                            if piece:
                                # If we're coming right after reasoning, delay text for a short hysteresis window
                                if thinking_open:
                                    no_reasoning_streak += 1
                                else:
                                    no_reasoning_streak = 0
                                if not sent_start:
                                    ms = ensure_message_start()
                                    if ms:
                                        yield ms
                                # If text arrives while thinking is open, we will only close thinking immediately
                                # when we have a structured tool call to emit; otherwise we buffer text until
                                # no_reasoning_streak reaches hysteresis to avoid mixing.
                                # Handle Cloud Code MCP tool-call markup robustly across stream chunks
                                BEG = "<|tool_calls_section_begin|>"
                                END = "<|tool_calls_section_end|>"
                                # Buffer for potential split markup across deltas
                                # Normalize potential fullwidth markers used by some models
                                try:
                                    norm_piece = (
                                        piece.replace("｜", "|")  # fullwidth vertical bar -> ASCII
                                        .replace("▁", "_")        # SentencePiece underscore glyph -> '_'
                                    )
                                except Exception:
                                    norm_piece = piece
                                mcp_buf += norm_piece  # type: ignore

                                def _parse_simple_calls(section: str):
                                    import re as _re
                                    simple_calls: list[dict] = []
                                    try:
                                        # Try multiple patterns to handle different character encodings
                                        patterns = [
                                            r"<\|tool_call_begin\|>(.*?)<\|tool_sep\|>(.*?)<\|tool_call_end\|>",
                                            r"<\|tool_call_begin\|>(.*?)<\|tool_sep\|>(.*?)(?:<\|tool_call_end\|>|\s*$)",
                                            r"<｜tool▁call▁begin｜>(.*?)<｜tool▁sep｜>(.*?)<｜tool▁call▁end｜>",
                                            r"<｜tool▁call▁begin｜>(.*?)<｜tool▁sep｜>(.*?)(?:<｜tool▁call▁end｜>|\s*$)"
                                        ]

                                        for pattern_str in patterns:
                                            try:
                                                pattern = _re.compile(pattern_str, _re.DOTALL)
                                                for m in pattern.finditer(section):
                                                    name = (m.group(1) or '').strip()
                                                    args_raw = (m.group(2) or '').strip()
                                                    try:
                                                        import json as _json
                                                        args = _json.loads(args_raw)
                                                    except Exception:
                                                        args = {}
                                                    simple_calls.append({
                                                        "type": "tool_use",
                                                        "id": f"call_{uuid.uuid4().hex}",
                                                        "name": name,
                                                        "input": args if isinstance(args, dict) else {}
                                                    })
                                                # If we found matches with this pattern, break
                                                if simple_calls:
                                                    break
                                            except Exception:
                                                continue
                                    except Exception as e:
                                        # Log the error for debugging but don't crash
                                        print(f"[proxy] Error parsing simple calls: {e}", file=sys.stderr)
                                        # Try alternative parsing for malformed tool calls
                                        try:
                                            # Handle case where tool calls might be malformed
                                            tools = ["TodoWrite", "Read", "Write", "Edit", "Grep", "Glob", "Bash"]
                                            for tool_name in tools:
                                                if tool_name in section:
                                                    # Try to extract tool calls with a more permissive pattern
                                                    alt_pattern = _re.compile(rf"{tool_name}[^\{{]*(\{{.*?\}})")
                                                    for m in alt_pattern.finditer(section):
                                                        try:
                                                            import json as _json
                                                            args = _json.loads(m.group(1))
                                                            simple_calls.append({
                                                                "type": "tool_use",
                                                                "id": f"call_{uuid.uuid4().hex}",
                                                                "name": tool_name,
                                                                "input": args if isinstance(args, dict) else {}
                                                            })
                                                        except Exception:
                                                            pass
                                        except Exception:
                                            pass
                                    return simple_calls

                                def _drain_mcp(buf: str):
                                    BEG1 = "<|tool_calls_section_begin|>"; END1 = "<|tool_calls_section_end|>"
                                    BEG2 = "<|tool_calls_begin|>"; END2 = "<|tool_calls_end|>"
                                    out_text_parts: list[str] = []
                                    calls: list[dict] = []
                                    pos = 0
                                    n = len(buf)
                                    while pos < n:
                                        i1 = buf.find(BEG1, pos); i2 = buf.find(BEG2, pos)
                                        candidates = [x for x in [i1, i2] if x != -1]
                                        if not candidates:
                                            out_text_parts.append(buf[pos:])
                                            break
                                        i = min(candidates)
                                        out_text_parts.append(buf[pos:i])
                                        if i == i1:
                                            j = buf.find(END1, i)
                                            if j == -1:
                                                return "".join(out_text_parts), calls, buf[i:]
                                            section = buf[i:j+len(END1)]
                                            norm_text, tcalls = parse_mcp_tool_markup(section)
                                            if norm_text:
                                                out_text_parts.append(norm_text)
                                            if tcalls:
                                                calls.extend(tcalls)
                                            pos = j + len(END1)
                                        else:
                                            j = buf.find(END2, i)
                                            if j == -1:
                                                return "".join(out_text_parts), calls, buf[i:]
                                            section = buf[i:j+len(END2)]
                                            tcalls = _parse_simple_calls(section)
                                            if tcalls:
                                                calls.extend(tcalls)
                                            pos = j + len(END2)
                                    return "".join(out_text_parts), calls, ""

                                # Drain buffer into plain text and tool calls when possible
                                drained_text, mcp_calls, mcp_left = _drain_mcp(mcp_buf)  # type: ignore
                                mcp_buf = mcp_left  # type: ignore
                                # Also drain unmarked calls like <ToolName>{...} from normal text stream (rare)
                                if mcp_buf:
                                    drained2, calls2, left2, moved_unknown = _drain_unmarked_calls(mcp_buf, known_tools, hold_tail_for_declared=False)
                                    if drained2:
                                        drained_text += drained2
                                    if calls2:
                                        mcp_calls.extend(calls2)
                                    mcp_buf = left2  # type: ignore
                                    if moved_unknown:
                                        # Unrecognized tool snippets: keep them as plain text (not in thinking)
                                        if not sent_start:
                                            ms = ensure_message_start()
                                            if ms:
                                                yield ms
                                        if text_block_index is None:
                                            text_block_index = next_block_index
                                            next_block_index += 1
                                            yield sse(
                                                "content_block_start",
                                                {
                                                    "type": "content_block_start",
                                                    "index": text_block_index,
                                                    "content_block": {"type": "text", "text": ""},
                                                },
                                            )
                                        yield sse(
                                            "content_block_delta",
                                            {
                                                "type": "content_block_delta",
                                                "index": text_block_index,
                                                "delta": {"type": "text_delta", "text": moved_unknown},
                                            },
                                        )

                                # First, emit any discovered tool calls from markup as tool_use blocks
                                for call in mcp_calls:
                                    try:
                                        import json as _json
                                        st_key = f"mcp_{len(tool_states)}"
                                        st = {
                                            "open": False,
                                            "cb_index": next_block_index,
                                            "id": call.get("id") or f"call_{uuid.uuid4().hex}",
                                            "name": _normalize_tool_name(call.get("name")) or "",
                                        }
                                        tool_states[st_key] = st
                                        cb_index = next_block_index
                                        next_block_index += 1
                                        yield sse(
                                            "content_block_start",
                                            {
                                                "type": "content_block_start",
                                                "index": cb_index,
                                                "content_block": {
                                                    "type": "tool_use",
                                                    "id": st["id"],
                                                    "name": st["name"],
                                                    "input": {},
                                                },
                                            },
                                        )
                                        st["open"] = True
                                        args_obj = call.get("input") or {}
                                        if st["open"] and args_obj:
                                            yield sse(
                                                "content_block_delta",
                                                {
                                                    "type": "content_block_delta",
                                                    "index": st["cb_index"],
                                                    "delta": {"type": "input_json_delta", "partial_json": _json.dumps(args_obj)},
                                                },
                                            )
                                    except Exception:
                                        ...

                                # Next, run sglang textual tool parser on remaining plain text
                                normal_piece = drained_text
                                if normal_piece and parser is not None:
                                    try:
                                        normal_piece, call_items = parser.parse_stream_chunk(normal_piece)
                                        for it in call_items or []:
                                            try:
                                                import json as _json
                                                fargs = _json.loads(it.parameters) if isinstance(it.parameters, str) else {}
                                            except Exception:
                                                fargs = {}
                                            mapped_name = _normalize_tool_name(it.name)
                                            st = tool_states.get(mapped_name or f"idx_{len(tool_states)}")
                                            if not st:
                                                st = tool_states[mapped_name or f"idx_{len(tool_states)}"] = {
                                                    "open": False,
                                                    "cb_index": next_block_index,
                                                    "id": f"call_{uuid.uuid4().hex}",
                                                    "name": mapped_name or "",
                                                }
                                                cb_index = next_block_index
                                                next_block_index += 1
                                                yield sse(
                                                    "content_block_start",
                                                    {
                                                        "type": "content_block_start",
                                                        "index": cb_index,
                                                        "content_block": {
                                                            "type": "tool_use",
                                                            "id": st["id"],
                                                            "name": st["name"],
                                                            "input": {},
                                                        },
                                                    },
                                                )
                                                st["open"] = True
                                            if st["open"] and fargs:
                                                yield sse(
                                                    "content_block_delta",
                                                    {
                                                        "type": "content_block_delta",
                                                        "index": st["cb_index"],
                                                        "delta": {"type": "input_json_delta", "partial_json": _json.dumps(fargs)},
                                                    },
                                                )
                                    except Exception:
                                        ...

                                # If there is buffered text carried over from a previous thinking chunk, emit it first
                                if pending_text_after_thinking:
                                    ts = ensure_text_block()
                                    if ts:
                                        yield ts
                                    yield sse(
                                        "content_block_delta",
                                        {
                                            "type": "content_block_delta",
                                            "index": text_block_index,
                                            "delta": {"type": "text_delta", "text": pending_text_after_thinking},
                                        },
                                    )
                                    pending_text_after_thinking = ""

                                # Finally, emit any remaining plain text
                                if normal_piece:
                                    if thinking_open and thinking_block_index is not None and not mcp_calls and no_reasoning_streak < hysteresis:
                                        # Defer text to avoid interleaving with thinking; accumulate and continue
                                        pending_text_after_thinking += normal_piece
                                        continue
                                    # Close thinking once we've waited sufficient chunks or when there were tool calls
                                    if thinking_open and thinking_block_index is not None:
                                        yield sse(
                                            "content_block_stop",
                                            {"type": "content_block_stop", "index": thinking_block_index},
                                        )
                                        thinking_open = False
                                        thinking_block_index = None
                                    ts = ensure_text_block()
                                    if ts:
                                        yield ts
                                    yield sse(
                                        "content_block_delta",
                                        {
                                            "type": "content_block_delta",
                                            "index": text_block_index,
                                            "delta": {"type": "text_delta", "text": normal_piece},
                                        },
                                    )

                            # Tool call deltas (support both delta.tool_calls and message.tool_calls)
                            tool_entries = delta.get("tool_calls") or ((choices[0].get("message") or {}).get("tool_calls") or [])
                            for tci in tool_entries:
                                idx = tci.get("index", 0)
                                st = tool_states.get(idx)
                                if st is None:
                                    # Initialize state but do not start until we have a name
                                    st = {
                                        "cb_index": None,
                                        "id": tci.get("id") or None,
                                        "name": None,
                                        "args": "",
                                        "open": False,
                                    }
                                    tool_states[idx] = st
                                # Capture id/name/args fragments
                                if tci.get("id"):
                                    st["id"] = tci.get("id")
                                fname = (tci.get("function") or {}).get("name")
                                if fname:
                                    fname_m = _normalize_tool_name(fname)
                                    st["name"] = (st.get("name") or "") + fname_m if not st.get("name") else st["name"]
                                fargs = (tci.get("function") or {}).get("arguments")
                                if fargs:
                                    st["args"] += fargs

                                # Start block only when we have a name
                                if not st["open"] and st.get("name"):
                                    if not sent_start:
                                        ms = ensure_message_start()
                                        if ms:
                                            yield ms
                                    cb_index = next_block_index
                                    next_block_index += 1
                                    st["cb_index"] = cb_index
                                    st["id"] = st.get("id") or f"call_{uuid.uuid4().hex}"
                                    yield sse(
                                        "content_block_start",
                                        {
                                            "type": "content_block_start",
                                            "index": cb_index,
                                            "content_block": {
                                                "type": "tool_use",
                                                "id": st["id"],
                                                "name": st["name"],
                                                "input": {},
                                            },
                                        },
                                    )
                                    st["open"] = True

                                # Stream arguments if present and block is open
                                if st["open"] and fargs:
                                    yield sse(
                                        "content_block_delta",
                                        {
                                            "type": "content_block_delta",
                                            "index": st["cb_index"],
                                            "delta": {"type": "input_json_delta", "partial_json": fargs},
                                        },
                                    )
                                    emitted_tool_use = True

                        # Capture usage on final chunk when present
                        usage = chunk.get("usage")
                        if usage:
                            usage_input = int(usage.get("prompt_tokens") or 0)
                            usage_output = int(usage.get("completion_tokens") or 0)

                        # If finish_reason present, finalize early
                        if choices and choices[0].get("finish_reason"):
                            fr = choices[0].get("finish_reason")
                            # Close blocks
                            # Flush any deferred text moved out from thinking before closing blocks
                            if pending_text_after_thinking:
                                if not sent_start:
                                    ms = ensure_message_start()
                                    if ms:
                                        yield ms
                                if text_block_index is None:
                                    text_block_index = next_block_index
                                    next_block_index += 1
                                    yield sse(
                                        "content_block_start",
                                        {"type": "content_block_start", "index": text_block_index, "content_block": {"type": "text", "text": ""}},
                                    )
                                yield sse(
                                    "content_block_delta",
                                    {
                                        "type": "content_block_delta",
                                        "index": text_block_index,
                                        "delta": {"type": "text_delta", "text": pending_text_after_thinking},
                                    },
                                )
                                pending_text_after_thinking = ""

                            if text_block_index is not None:
                                yield sse("content_block_stop", {"type": "content_block_stop", "index": text_block_index})
                                text_block_index = None
                            if thinking_open and thinking_block_index is not None:
                                yield sse("content_block_stop", {"type": "content_block_stop", "index": thinking_block_index})
                                thinking_open = False
                                thinking_block_index = None
                            for st in list(tool_states.values()):
                                if st.get("open"):
                                    yield sse("content_block_stop", {"type": "content_block_stop", "index": st["cb_index"]})
                                    st["open"] = False
                            # message_delta
                            final_stop = "tool_use" if emitted_tool_use else map_finish_reason(fr)
                            yield sse(
                                "message_delta",
                                {
                                    "type": "message_delta",
                                    "delta": {"stop_reason": final_stop, "stop_sequence": None},
                                    "usage": {"input_tokens": usage_input, "output_tokens": usage_output},
                                },
                            )
                            yield sse("message_stop", {"type": "message_stop"})
                            _rec["phase"] = "stream_ok_finish"
                            _RECENT.append(_rec)
                            break
                # After stream finishes, finalize depending on whether any chunks arrived
                if no_chunks:
                    # Instead of throwing an error, gracefully handle empty streams by sending an empty response
                    # This can happen with certain models or under specific conditions
                    if not sent_start:
                        ms = ensure_message_start()
                        if ms:
                            yield ms

                    # Emit final message delta and stop with end_turn reason for empty responses
                    yield sse(
                        "message_delta",
                        {
                            "type": "message_delta",
                            "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                            "usage": {"input_tokens": usage_input, "output_tokens": usage_output},
                        },
                    )
                    yield sse("message_stop", {"type": "message_stop"})
                    _rec["phase"] = "stream_empty_no_chunks_handled"
                    _RECENT.append(_rec)
                else:
                    # Upstream ended without explicit terminator ([DONE] / finish_reason). Gracefully close blocks and stop.
                    # Flush any deferred text moved out from thinking before closing blocks
                    if pending_text_after_thinking:
                        if not sent_start:
                            ms = ensure_message_start()
                            if ms:
                                yield ms
                        if text_block_index is None:
                            text_block_index = next_block_index
                            next_block_index += 1
                            yield sse(
                                "content_block_start",
                                {"type": "content_block_start", "index": text_block_index, "content_block": {"type": "text", "text": ""}},
                            )
                        yield sse(
                            "content_block_delta",
                            {
                                "type": "content_block_delta",
                                "index": text_block_index,
                                "delta": {"type": "text_delta", "text": pending_text_after_thinking},
                            },
                        )
                        pending_text_after_thinking = ""

                    # Close any open content blocks
                    if text_block_index is not None:
                        yield sse("content_block_stop", {"type": "content_block_stop", "index": text_block_index})
                        text_block_index = None
                    if thinking_open and thinking_block_index is not None:
                        yield sse("content_block_stop", {"type": "content_block_stop", "index": thinking_block_index})
                        thinking_open = False
                        thinking_block_index = None
                    for st in list(tool_states.values()):
                        if st.get("open"):
                            yield sse("content_block_stop", {"type": "content_block_stop", "index": st["cb_index"]})
                            st["open"] = False

                    # Emit final message delta and stop with a conservative stop_reason
                    final_stop = "tool_use" if emitted_tool_use else "end_turn"
                    yield sse(
                        "message_delta",
                        {
                            "type": "message_delta",
                            "delta": {"stop_reason": final_stop, "stop_sequence": None},
                            "usage": {"input_tokens": usage_input, "output_tokens": usage_output},
                        },
                    )
                    yield sse("message_stop", {"type": "message_stop"})
                    _rec["phase"] = "stream_ok_eof"
                    _RECENT.append(_rec)
                    print("[proxy] Stream ended normally", file=sys.stderr)
        except Exception as e:
            # Surface upstream error as Anthropic error (single event) with graceful stop
            msx = ensure_message_start()
            if msx:
                yield msx
            # Log the exception for debugging purposes
            print(f"[proxy] stream exception: {type(e).__name__}: {str(e)}", file=sys.stderr)
            import traceback
            traceback.print_exc(file=sys.stderr)

            data = {"type": "error", "error": {"type": "upstream_error", "message": str(e)}}
            yield f"event: error\n".encode() + f"data: {json.dumps(data)}\n\n".encode()
            yield sse("message_stop", {"type": "message_stop"})
            _rec["phase"] = "stream_exception"
            _rec["message"] = str(e)
            _rec["exception_type"] = type(e).__name__
            _RECENT.append(_rec)
            print(f"[proxy] Stream ended with exception: {type(e).__name__}: {str(e)}", file=sys.stderr)

    return StreamingResponse(event_stream(request), media_type="text/event-stream")


@app.get("/")
async def root():
    return {"ok": True, "backend": settings.chutes_base_url}


@app.get("/v1/models")
async def list_models(
    x_api_key: str | None = Header(default=None, alias="x-api-key"),
    authorization: str | None = Header(default=None, alias="authorization"),
):
    # Use cached model discovery to avoid frequent upstream calls
    headers = _auth_headers(x_api_key, authorization)
    model_ids = await _get_model_ids(headers)
    
    if model_ids is None:
        # If cache miss and upstream fetch fails, try direct upstream call as fallback
        url = f"{settings.chutes_base_url.rstrip('/')}/v1/models"
        client = _get_httpx_client()
        try:
            resp = await client.get(url, headers=headers, timeout=httpx.Timeout(60.0))
        except Exception as e:
            raise HTTPException(status_code=502, detail=f"Upstream error: {e}")
        if resp.status_code >= 400:
            try:
                data = resp.json()
                message = data.get("error", {}).get("message") or data.get("message") or json.dumps(data)
            except Exception:
                message = resp.text
            return JSONResponse(status_code=resp.status_code, content=_anthropic_error_for_status(resp.status_code, message))
        return JSONResponse(content=resp.json())
    
    # Format cached model ids to match upstream API response structure
    models_data = [{"id": model_id, "object": "model", "created": 0, "owned_by": "chutes"} for model_id in model_ids]
    response_data = {
        "object": "list",
        "data": models_data
    }
    return JSONResponse(content=response_data)


@app.get("/_debug/last")
async def debug_last():
    return _RECENT[-1] if _RECENT else {}


@app.on_event("startup")
async def _startup_noop():
    # Initialize shared HTTP client eagerly to establish pools
    _ = _get_httpx_client()
    return None


@app.on_event("shutdown")
async def _shutdown_close_client():
    global _HTTPX_CLIENT
    if _HTTPX_CLIENT is not None:
        try:
            await _HTTPX_CLIENT.aclose()
        except Exception:
            ...
        _HTTPX_CLIENT = None


# Admin: view current models cache entry (for this upstream + auth fingerprint)
@app.get("/_models_cache")
async def get_models_cache(
    x_api_key: str | None = Header(default=None, alias="x-api-key"),
    authorization: str | None = Header(default=None, alias="authorization"),
):
    headers = _auth_headers(x_api_key, authorization)
    key = _models_cache_key(headers)
    now = asyncio.get_event_loop().time()
    # in-memory
    ent: Dict[str, Any] = {}
    if key in _MODEL_LIST_CACHE:
        ts, ids = _MODEL_LIST_CACHE[key]
        ent = {"ids": ids, "ts": int(ts), "source": "memory"}
    # disk
    if settings.model_discovery_persist and os.path.exists(settings.model_cache_file):
        try:
            with open(settings.model_cache_file, "r", encoding="utf-8") as f:
                obj = json.load(f) or {}
            dent = obj.get(key)
            if dent:
                ent = {**dent, "source": dent.get("source") or ent.get("source") or "disk"}
        except Exception:
            ...
    ent.setdefault("base_url", settings.chutes_base_url)
    ent.setdefault("now", int(now))
    ent.setdefault("key", key)
    return ent


# Admin: force refresh models and persist
@app.post("/_models_cache/refresh")
async def refresh_models_cache(
    x_api_key: str | None = Header(default=None, alias="x-api-key"),
    authorization: str | None = Header(default=None, alias="authorization"),
):
    headers = _auth_headers(x_api_key, authorization)
    ids = await _get_model_ids(headers)
    if not ids:
        raise HTTPException(status_code=502, detail="Failed to fetch models from upstream")
    return {"ok": True, "count": len(ids)}


# Admin: clear models cache entry
@app.delete("/_models_cache")
async def clear_models_cache(
    x_api_key: str | None = Header(default=None, alias="x-api-key"),
    authorization: str | None = Header(default=None, alias="authorization"),
):
    headers = _auth_headers(x_api_key, authorization)
    key = _models_cache_key(headers)
    removed = False
    if key in _MODEL_LIST_CACHE:
        _MODEL_LIST_CACHE.pop(key, None)
        removed = True
    if settings.model_discovery_persist and os.path.exists(settings.model_cache_file):
        try:
            async with _MODEL_CACHE_LOCK:
                with open(settings.model_cache_file, "r", encoding="utf-8") as f:
                    obj = json.load(f) or {}
                if key in obj:
                    obj.pop(key)
                    removed = True
                with open(settings.model_cache_file, "w", encoding="utf-8") as f:
                    json.dump(obj, f, ensure_ascii=False, indent=2)
        except Exception:
            ...
    return {"ok": True, "removed": removed}
