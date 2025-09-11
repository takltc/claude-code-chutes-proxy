from __future__ import annotations

import time
import uuid
from typing import Any, Dict, List, Optional
import re
import uuid

# Use sglang's OpenAI protocol models for validation/normalization
try:
    from sglang.srt.entrypoints.openai.protocol import (
        ChatCompletionRequest,
        ChatCompletionResponse,
    )
except Exception:  # pragma: no cover - validation remains optional fallback
    ChatCompletionRequest = None  # type: ignore
    ChatCompletionResponse = None  # type: ignore

# Defer FunctionCallParser import to runtime to avoid optional deps issues
FunctionCallParser = None  # type: ignore

from .config import settings
# schema_registry was used for custom OpenAI model capability tweaks. Removed in favor of sglang-only OpenAI side.


def _to_text(content) -> str:
    """Collapse Anthropic content (string or list of text blocks) into a plain string."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    # list of blocks
    parts: List[str] = []
    for block in content:
        if isinstance(block, dict):
            if block.get("type") == "text":
                parts.append(block.get("text", ""))
        else:
            # pydantic model
            t = getattr(block, "type", None)
            if t == "text":
                parts.append(getattr(block, "text", ""))
    return "".join(parts)


def anthropic_to_openai_payload(body: Dict[str, Any]) -> Dict[str, Any]:
    """Map Anthropic v1/messages body to OpenAI Chat Completions for Chutes backend."""
    model = settings.map_model(body.get("model"))
    messages = body.get("messages", [])
    system = body.get("system")

    oai_messages: List[Dict[str, Any]] = []
    # Map `system` to a system message if present
    if system:
        oai_messages.append({"role": "system", "content": _to_text(system)})

    # We must normalize tool_result/tool_use blocks into OpenAI-style messages
    pending_user_text: Optional[str] = None
    for m in messages:
        role = m.get("role")
        content = m.get("content")
        if role not in ("user", "assistant", "system"):
            role = "user"

        # Plain string content short-circuit
        if isinstance(content, str):
            oai_messages.append({"role": role, "content": content})
            continue

        # content is a list of blocks
        text_acc: List[str] = []
        image_items: List[Dict[str, Any]] = []
        has_multimodal = False
        tool_calls: List[Dict[str, Any]] = []
        tool_results: List[Dict[str, Any]] = []
        for block in content or []:
            btype = block.get("type")
            if btype == "text":
                text_acc.append(block.get("text", ""))
            elif btype == "image":
                # Anthropic image block formats: {type: 'image', source: {type: 'base64', media_type, data}} or {type:'image', source:{type:'url', url}}
                src = block.get("source") or {}
                url = None
                if (src.get("type") or "").lower() in ("base64", "base64_image", "base64jpeg", "base64png"):
                    media_type = src.get("media_type") or "image/png"
                    data = src.get("data") or ""
                    if data:
                        url = f"data:{media_type};base64,{data}"
                elif (src.get("type") or "").lower() in ("url", "external"):
                    url = src.get("url")
                if url:
                    image_items.append({"type": "image_url", "image_url": {"url": url}})
                    has_multimodal = True
            elif btype == "tool_use" and role == "assistant":
                # Convert to OpenAI tool_call entry
                call_id = block.get("id") or f"call_{uuid.uuid4().hex}"
                tool_calls.append(
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {
                            "name": block.get("name", ""),
                            "arguments": json_dumps_safe(block.get("input", {})),
                        },
                    }
                )
            elif btype == "tool_result" and role == "user":
                # Will become a separate 'tool' role message later
                tool_results.append(block)
            # other block types ignored

        text_content = "".join(text_acc)

        if role == "assistant":
            msg: Dict[str, Any] = {"role": "assistant"}
            if text_content:
                msg["content"] = text_content
            else:
                msg["content"] = ""
            if tool_calls:
                msg["tool_calls"] = tool_calls
            oai_messages.append(msg)
        elif role in ("user", "system"):
            # First, push user/system text/images (if any)
            if has_multimodal or image_items:
                content_list: List[Dict[str, Any]] = []
                if text_content:
                    content_list.append({"type": "text", "text": text_content})
                content_list.extend(image_items)
                if content_list:
                    oai_messages.append({"role": role, "content": content_list})
            elif text_content or role == "system":
                oai_messages.append({"role": role, "content": text_content})
            # Then, push tool results as separate 'tool' messages
            for tr in tool_results:
                oai_messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tr.get("tool_use_id"),
                        "content": _to_text(tr.get("content")),
                        "name": tr.get("name"),
                    }
                )

    payload: Dict[str, Any] = {
        "model": model,
        "messages": oai_messages,
        "max_tokens": body.get("max_tokens"),
        "stream": bool(body.get("stream")),
    }
    # Optional params
    if body.get("temperature") is not None:
        payload["temperature"] = body.get("temperature")
    if body.get("top_p") is not None:
        payload["top_p"] = body.get("top_p")
    if body.get("stop_sequences"):
        payload["stop"] = body.get("stop_sequences")

    # Tools definitions mapping (Anthropic -> OpenAI)
    tools = body.get("tools")
    if tools:
        oai_tools = []
        for t in tools:
            oai_tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": t.get("name"),
                        "description": t.get("description"),
                        "parameters": t.get("input_schema", {}),
                    },
                }
            )
        payload["tools"] = oai_tools
        # Encourage upstream function-calling
        payload["tool_choice"] = "auto"
    # Validate and normalize via sglang's OpenAI schema if available
    if ChatCompletionRequest is not None:
        try:
            req = ChatCompletionRequest.model_validate(payload)
            payload = req.model_dump(exclude_none=True)
        except Exception:
            # Keep original payload on validation errors to be permissive
            ...

    return payload


## Removed custom model capability adaptation. OpenAI-side behavior is determined by sglang/OpenAI schema.


def parse_mcp_tool_markup(text: str) -> tuple[str, list[Dict[str, Any]]]:
    """Parse Cloud Code MCP tool-call markup into structured tool_use blocks.

    Recognizes sections of the form:
      <|tool_calls_section_begin|>
        <|tool_call_begin|>functions.mcp__server__tool:1<|tool_call_argument_begin|>{...}<|tool_call_end|>
      <|tool_calls_section_end|>

    Returns (remaining_text_without_markup, tool_use_blocks)
    """
    if not text:
        return text, []

    BEG = "<|tool_calls_section_begin|>"
    END = "<|tool_calls_section_end|>"
    CALL_RE = re.compile(
        r"<\|tool_call_begin\|>(.*?)<\|tool_call_argument_begin\|>(.*?)<\|tool_call_end\|>",
        re.DOTALL,
    )

    out_text_parts: list[str] = []
    calls: list[Dict[str, Any]] = []
    pos = 0
    n = len(text)

    while pos < n:
        i = text.find(BEG, pos)
        if i == -1:
            out_text_parts.append(text[pos:])
            break
        # text before the section
        out_text_parts.append(text[pos:i])
        j = text.find(END, i)
        if j == -1:
            # No closing tag; keep as plain text
            out_text_parts.append(text[i:])
            break
        section = text[i + len(BEG) : j]
        for m in CALL_RE.finditer(section):
            full_name = (m.group(1) or "").strip()
            args_raw = (m.group(2) or "").strip()
            # Strip trailing :index and optional namespace prefix like "functions."
            name = full_name
            if name.startswith("functions.") or name.startswith("tools."):
                name = name.split(".", 1)[1]
            m_idx = re.match(r"^(.*?):\d+$", name)
            if m_idx:
                name = m_idx.group(1)
            try:
                args = json_loads_safe(args_raw)
            except Exception:
                args = {}
            calls.append(
                {
                    "type": "tool_use",
                    "id": f"call_{uuid.uuid4().hex}",
                    "name": name,
                    "input": args if isinstance(args, dict) else {},
                }
            )
        # Skip the whole section in output text
        pos = j + len(END)

    return "".join(out_text_parts), calls


def choose_tool_call_parser(model_name: Optional[str]) -> str:
    s = (model_name or "").lower()
    if not s:
        return "pythonic"
    # Longcat models frequently adopt T4/GPT‑OSS style tool-call channels
    # e.g., <|channel|>commentary to=functions.xxx<|constrain|>json<|message|>{...}<|call|>
    # Map them to the sglang "gpt-oss" detector.
    if "longcat" in s:
        return "gpt-oss"
    if "llama-3" in s or "llama3" in s:
        return "llama3"
    if "qwen3" in s:
        return "qwen3_coder"
    if "qwen" in s:
        # default to 2.5-style
        return "qwen25"
    if "mistral" in s:
        return "mistral"
    if "deepseek" in s:
        if "3.1" in s or "v3.1" in s or "v31" in s:
            return "deepseekv31"
        return "deepseekv3"
    if "kimi" in s or "k2" in s:
        return "kimi_k2"
    if "glm-4.5" in s or "glm45" in s or "glm4.5" in s:
        return "glm45"
    if "gpt-oss" in s or "gpt_oss" in s:
        return "gpt-oss"
    if "step-3" in s or "step3" in s or s.startswith("o3") or "-o3" in s:
        return "step3"
    return "pythonic"


def openai_to_anthropic_response(
    oai: Dict[str, Any],
    requested_model: Optional[str] = None,
    tools: Optional[List[Dict[str, Any]]] = None,
    tool_call_parser: Optional[str] = None,
) -> Dict[str, Any]:
    """Map a non-streaming OpenAI Chat Completions response to Anthropic messages response."""
    # Extract content
    text = ""
    tool_calls_block: List[Dict[str, Any]] = []
    def _maybe_parse_tools_with_sglang(text_block: str) -> tuple[str, list[Dict[str, Any]]]:
        global FunctionCallParser
        if FunctionCallParser is None:
            try:
                from sglang.srt.function_call.function_call_parser import (
                    FunctionCallParser as _FCP,
                )
                FunctionCallParser = _FCP  # type: ignore
            except Exception:
                FunctionCallParser = None  # type: ignore
        if not text_block or not tools or FunctionCallParser is None:
            return text_block or "", []

        # Build minimal Tool-like objects acceptable by FunctionCallParser
        class _Fn:
            def __init__(self, name: str, parameters: Dict[str, Any], strict: bool = False):
                self.name = name
                self.parameters = parameters
                self.strict = strict

        class _Tool:
            def __init__(self, fn: _Fn):
                self.function = fn

        tool_objs = []
        for t in tools or []:
            fn = (t or {}).get("function") or {}
            name = fn.get("name") or ""
            params = fn.get("parameters") or {}
            strict = bool(fn.get("strict", False))
            if name:
                tool_objs.append(_Tool(_Fn(name=name, parameters=params, strict=strict)))
        if not tool_objs:
            return text_block, []

        try:
            parser_name = tool_call_parser or choose_tool_call_parser(requested_model or oai.get("model"))
            parser = FunctionCallParser(tool_objs, parser_name)
            normal_text, call_items = parser.parse_non_stream(text_block)
            out_calls: list[Dict[str, Any]] = []
            for i, it in enumerate(call_items or []):
                try:
                    import json as _json
                    args = _json.loads(it.parameters) if isinstance(it.parameters, str) else {}
                except Exception:
                    args = {}
                out_calls.append(
                    {
                        "type": "tool_use",
                        "id": f"call_{uuid.uuid4().hex}",
                        "name": it.name or "",
                        "input": args,
                    }
                )
            return normal_text, out_calls
        except Exception:
            return text_block, []

    # No model-specific textual parsing; rely on sglang parser only

    # Optional: validate upstream response shape
    if ChatCompletionResponse is not None:
        try:
            _ = ChatCompletionResponse.model_validate(oai)
        except Exception:
            ...

    try:
        first_choice = (oai.get("choices") or [{}])[0]
        message = first_choice.get("message") or {}
        text = message.get("content") or ""
        # If available, include reasoning content as a dedicated thinking block (for proper UI display)
        reasoning = message.get("reasoning_content")
        reasoning_block = []
        if reasoning:
            reasoning_block = [
                {
                    "type": "thinking",
                    "thinking": str(reasoning),
                }
            ]
        # Convert OpenAI tool_calls to Anthropic tool_use blocks
        tcalls = message.get("tool_calls") or []
        for tc in tcalls:
            try:
                args = json_loads_safe((tc.get("function") or {}).get("arguments"))
            except Exception:
                args = {}
            tool_calls_block.append(
                {
                    "type": "tool_use",
                    "id": tc.get("id") or f"call_{uuid.uuid4().hex}",
                    "name": (tc.get("function") or {}).get("name"),
                    "input": args,
                }
            )
        # Parse Cloud Code MCP tool-call markup first
        if text:
            text, markup_calls = parse_mcp_tool_markup(text)
            tool_calls_block.extend(markup_calls)
        # Fallback: use sglang parser on textual content when no structured tool_calls
        if not tcalls and text:
            text, parsed_calls = _maybe_parse_tools_with_sglang(text)
            tool_calls_block.extend(parsed_calls)
    except Exception:
        text = ""

    finish_reason = None
    try:
        fr = (oai.get("choices") or [{}])[0].get("finish_reason")
    except Exception:
        fr = None

    anthropic_stop_reason = map_finish_reason(fr)

    usage = oai.get("usage") or {}

    content_blocks: List[Dict[str, Any]] = []
    if text:
        if str(text).strip():
            content_blocks.append({"type": "text", "text": text})
    # Prepend reasoning if present (as a thinking block)
    if 'reasoning_block' in locals() and reasoning_block:
        content_blocks = reasoning_block + content_blocks
    content_blocks.extend(tool_calls_block)

    return {
        "id": f"msg_{uuid.uuid4().hex}",
        "type": "message",
        "role": "assistant",
        "model": requested_model or oai.get("model", "unknown-model"),
        "content": content_blocks or [{"type": "text", "text": ""}],
        "stop_reason": anthropic_stop_reason,
        "stop_sequence": None,
        "usage": {
            "input_tokens": int(usage.get("prompt_tokens") or 0),
            "output_tokens": int(usage.get("completion_tokens") or 0),
        },
    }


def parse_stream_tools_text(
    text_chunk: str,
    tools: Optional[List[Dict[str, Any]]],
    model_name: Optional[str] = None,
) -> tuple[str, list[Dict[str, Any]]]:
    """Parse a streaming text chunk for tool calls using sglang's FunctionCallParser.

    Returns (normal_text, tool_use_blocks)
    """
    if not text_chunk or not tools:
        return text_chunk or "", []
    # No model-specific textual branch; rely on sglang parser below

    global FunctionCallParser
    if FunctionCallParser is None:
        try:
            from sglang.srt.function_call.function_call_parser import (
                FunctionCallParser as _FCP,
            )
            FunctionCallParser = _FCP  # type: ignore
        except Exception:
            return text_chunk, []

    class _Fn:
        def __init__(self, name: str, parameters: Dict[str, Any], strict: bool = False):
            self.name = name
            self.parameters = parameters
            self.strict = strict

    class _Tool:
        def __init__(self, fn: _Fn):
            self.function = fn

    tool_objs = []
    for t in tools or []:
        fn = (t or {}).get("function") or {}
        name = fn.get("name") or ""
        params = fn.get("parameters") or {}
        strict = bool(fn.get("strict", False))
        if name:
            tool_objs.append(_Tool(_Fn(name=name, parameters=params, strict=strict)))
    if not tool_objs:
        return text_chunk, []

    try:
        parser_name = choose_tool_call_parser(model_name)
        parser = FunctionCallParser(tool_objs, parser_name)
        normal_text, call_items = parser.parse_stream_chunk(text_chunk)
        calls: list[Dict[str, Any]] = []
        for it in call_items or []:
            try:
                import json as _json
                args = _json.loads(it.parameters) if isinstance(it.parameters, str) else {}
            except Exception:
                args = {}
            calls.append(
                {
                    "type": "tool_use",
                    "id": f"call_{uuid.uuid4().hex}",
                    "name": it.name or "",
                    "input": args,
                }
            )
        return normal_text, calls
    except Exception:
        return text_chunk, []

    # Optional: validate upstream response shape
    if ChatCompletionResponse is not None:
        try:
            _ = ChatCompletionResponse.model_validate(oai)
        except Exception:
            ...

    try:
        first_choice = (oai.get("choices") or [{}])[0]
        message = first_choice.get("message") or {}
        text = message.get("content") or ""
        # If available, include reasoning content as separate block prefix
        reasoning = message.get("reasoning_content")
        reasoning_block = []
        if reasoning:
            reasoning_block = [{"type": "text", "text": str(reasoning)}]
        # Convert OpenAI tool_calls to Anthropic tool_use blocks
        tcalls = message.get("tool_calls") or []
        for tc in tcalls:
            try:
                args = json_loads_safe((tc.get("function") or {}).get("arguments"))
            except Exception:
                args = {}
            tool_calls_block.append(
                {
                    "type": "tool_use",
                    "id": tc.get("id") or f"call_{uuid.uuid4().hex}",
                    "name": (tc.get("function") or {}).get("name"),
                    "input": args,
                }
            )
        # Fallback: use sglang tool parser on textual content when no structured tool_calls
        if not tcalls and text:
            text, parsed_calls = _maybe_parse_tools_with_sglang(text)
            tool_calls_block.extend(parsed_calls)
    except Exception:
        text = ""

    finish_reason = None
    try:
        fr = (oai.get("choices") or [{}])[0].get("finish_reason")
    except Exception:
        fr = None

    anthropic_stop_reason = map_finish_reason(fr)

    usage = oai.get("usage") or {}

    content_blocks: List[Dict[str, Any]] = []
    if text:
        if str(text).strip():
            content_blocks.append({"type": "text", "text": text})
    # Prepend reasoning if present
    if 'reasoning_block' in locals() and reasoning_block:
        content_blocks = reasoning_block + content_blocks
    content_blocks.extend(tool_calls_block)

    return {
        "id": f"msg_{uuid.uuid4().hex}",
        "type": "message",
        "role": "assistant",
        "model": requested_model or oai.get("model", "unknown-model"),
        "content": content_blocks or [{"type": "text", "text": ""}],
        "stop_reason": anthropic_stop_reason,
        "stop_sequence": None,
        "usage": {
            "input_tokens": int(usage.get("prompt_tokens") or 0),
            "output_tokens": int(usage.get("completion_tokens") or 0),
        },
    }


def map_finish_reason(fr: Optional[str]) -> Optional[str]:
    if fr is None:
        return None
    if fr == "stop":
        return "end_turn"
    if fr == "length":
        return "max_tokens"
    if fr == "content_filter":
        # No perfect mapping; return end_turn
        return "end_turn"
    if fr == "tool_calls":
        return "tool_use"
    return "end_turn"


def now_unix() -> int:
    return int(time.time())


def json_dumps_safe(obj: Any) -> str:
    try:
        import json

        return json.dumps(obj, ensure_ascii=False)
    except Exception:
        return "{}"


def json_loads_safe(s: Any) -> Any:
    try:
        import json

        if isinstance(s, str):
            return json.loads(s)
        return s
    except Exception:
        return {}
