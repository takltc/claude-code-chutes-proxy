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
    requested_model = body.get("model")
    model = settings.map_model(requested_model)
    messages = body.get("messages", [])
    system = body.get("system")
    system_text = _to_text(system)
    longcat_mode = is_longcat_model(requested_model) or is_longcat_model(model)

    oai_messages: List[Dict[str, Any]] = []
    # Map `system` to a system message if present (non-LongCat requests; LongCat overrides later)
    if system_text:
        oai_messages.append({"role": "system", "content": system_text})

    # We must normalize tool_result/tool_use blocks into OpenAI-style messages
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

    if longcat_mode:
        prompt_text = format_longcat_prompt(system, messages, tools)
        payload["messages"] = [{"role": "user", "content": prompt_text}]
    # Validate and normalize via sglang's OpenAI schema if available
    if ChatCompletionRequest is not None:
        try:
            req = ChatCompletionRequest.model_validate(payload)
            payload = req.model_dump(exclude_none=True)
        except Exception:
            # Keep original payload on validation errors to be permissive
            ...

    return payload


def format_deepseek_v31_tool_call_payload(
    system_prompt: str, tools: List[Dict[str, Any]], query: str
) -> Dict[str, Any]:
    """
    Format the payload for DeepSeek V3.1 tool calls according to the specified format.

    Toolcall format: <｜begin▁of▁sentence｜>{system prompt}\n\n{tool_description}<｜User｜>{query}<｜Assistant｜></think>
    """
    # Build tool description according to the format
    tool_description = "## Tools\nYou have access to the following tools:\n\n"
    for tool in tools:
        tool_description += f"### {tool.get('function', {}).get('name', '')}\n"
        tool_description += f"Description: {tool.get('function', {}).get('description', '')}\n"
        tool_description += f"Parameters: {json_dumps_safe(tool.get('function', {}).get('parameters', {}))}\n\n"
    
    # Format the prompt according to DeepSeek V3.1 requirements
    formatted_prompt = (
        f"<｜begin▁of▁sentence｜>{system_prompt}\n\n"
        f"{tool_description}<｜User｜>{query}<｜Assistant｜></think>"
    )
    
    return {
        "prompt": formatted_prompt,
        "tool_description": tool_description
    }


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


def is_longcat_model(model_name: Optional[str]) -> bool:
    if not model_name:
        return False
    try:
        return "longcat" in str(model_name).lower()
    except Exception:
        return False


def choose_tool_call_parser(model_name: Optional[str]) -> str:
    s = (model_name or "").lower()
    if not s:
        return "pythonic"
    # Longcat models frequently adopt T4/GPT‑OSS style tool-call channels
    # e.g., <|channel|>commentary to=functions.xxx<|constrain|>json<|message|>{...}<|call|>
    # Map them to the sglang "gpt-oss" detector.
    if is_longcat_model(s):
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


def _block_get(block: Any, key: str, default: Any = None) -> Any:
    if isinstance(block, dict):
        return block.get(key, default)
    return getattr(block, key, default)


def _longcat_dump_json(data: Any, indent: int = 2) -> str:
    try:
        import json

        return json.dumps(data, ensure_ascii=False, indent=indent)
    except Exception:
        return json_dumps_safe(data)


def _flatten_longcat_text(text: Optional[str]) -> str:
    if not text:
        return ""
    stripped = str(text).strip()
    if not stripped:
        return ""
    if "<longcat_tool_call>" in stripped or stripped.startswith("{") or stripped.startswith("["):
        return stripped
    return " ".join(stripped.split())


def _longcat_message_text(message: Dict[str, Any]) -> str:
    if not message:
        return ""
    content = message.get("content")
    role = message.get("role")
    if isinstance(content, str):
        return content.strip()
    parts: List[str] = []
    for block in content or []:
        btype = _block_get(block, "type")
        if btype == "text":
            parts.append(str(_block_get(block, "text", "")))
        elif btype == "tool_use" and role == "assistant":
            payload = {
                "name": _block_get(block, "name", ""),
                "arguments": _block_get(block, "input", {}) or {},
            }
            parts.append(
                "<longcat_tool_call>\n"
                f"{_longcat_dump_json(payload)}\n"
                "</longcat_tool_call>"
            )
        elif btype == "tool_result":
            name = _block_get(block, "name", "") or ""
            result = _to_text(_block_get(block, "content"))
            if result:
                label = f"[tool_result name={name}]" if name else "[tool_result]"
                parts.append(f"{label} {result}")
        elif btype == "image":
            source = _block_get(block, "source") or {}
            media_type = _block_get(source, "media_type", "image") if isinstance(source, dict) else getattr(source, "media_type", "image")
            parts.append(f"[image {media_type}]")
    return "\n".join(p.strip() for p in parts if p is not None)


def _merge_longcat_strings(first: Optional[str], second: Optional[str]) -> str:
    first = first or ""
    second = second or ""
    if not first:
        return second
    if not second:
        return first
    return f"{first}\n{second}"


def _longcat_tool_description(tools: Optional[List[Dict[str, Any]]]) -> str:
    parts: List[str] = ["markdown"]
    if not tools:
        return "\n".join(parts)
    parts.extend(["", "## Tools", "You have access to the following tools:", ""])
    for tool in tools:
        function = (tool or {}).get("function") or {}
        name = (function.get("name") or tool.get("name") or "").strip()
        description = (function.get("description") or tool.get("description") or "").strip()
        parameters = function.get("parameters") or tool.get("parameters") or {}
        schema = _longcat_dump_json(parameters, indent=2)
        parts.append("### Tool namespace: function")
        parts.append("")
        parts.append(f"#### Tool name: {name}")
        parts.append("")
        parts.append(f"Description: {description}")
        parts.append("")
        parts.append("InputSchema:")
        parts.append(schema)
        parts.append("")
    return "\n".join(parts).rstrip()


def format_longcat_prompt(
    system_prompt: Optional[str],
    messages: List[Dict[str, Any]],
    tools: Optional[List[Dict[str, Any]]] = None,
) -> str:
    system_parts: List[str] = []
    if system_prompt:
        system_text = _flatten_longcat_text(_to_text(system_prompt))
        if system_text:
            system_parts.append(system_text)

    rounds: List[Dict[str, Optional[str]]] = []
    for message in messages or []:
        role = message.get("role")
        text = _longcat_message_text(message)
        if role == "system":
            if text:
                system_parts.append(_flatten_longcat_text(text))
            continue
        if role == "assistant":
            if not rounds:
                rounds.append({"user": "", "assistant": text})
            elif rounds[-1].get("assistant") is None:
                rounds[-1]["assistant"] = text
            else:
                rounds.append({"user": "", "assistant": text})
            continue
        if not rounds:
            rounds.append({"user": text, "assistant": None})
            continue
        if rounds[-1].get("assistant") is None:
            rounds[-1]["user"] = _merge_longcat_strings(rounds[-1].get("user"), text)
        else:
            rounds.append({"user": text, "assistant": None})

    if not rounds:
        rounds.append({"user": "", "assistant": None})

    system_line = " ".join(part for part in system_parts if part).strip()
    lines: List[str] = []
    if system_line:
        lines.append(f"SYSTEM:{system_line}")

    total_rounds = len(rounds)
    for idx, round_entry in enumerate(rounds):
        user_text = _flatten_longcat_text(round_entry.get("user") or "")
        assistant_raw = round_entry.get("assistant")
        if assistant_raw is None and idx == total_rounds - 1:
            lines.append(f"[Round {idx}] USER:{user_text} ASSISTANT:")
        else:
            assistant_text = _flatten_longcat_text(assistant_raw or "")
            lines.append(f"[Round {idx}] USER:{user_text} ASSISTANT:{assistant_text}</longcat_s>")

    messages_section = "\n".join(lines).strip()
    tool_section = _longcat_tool_description(tools)
    sections = [section for section in [tool_section, "## Messages", messages_section] if section]
    return "\n\n".join(sections).strip()


def format_deepseek_v31_prompt(
    system_prompt: str, user_query: str, context: str = "", is_thinking: bool = False
) -> str:
    """
    Format prompt according to DeepSeek V3.1 requirements.

    Non-thinking mode prefix: <｜begin▁of▁sentence｜>{system prompt}<｜User｜>{query}<｜Assistant｜></think>
    Non-thinking mode context: <｜begin▁of▁sentence｜>{system prompt}<｜User｜>{query}<｜Assistant｜></think>{response}<｜end▁of▁sentence｜>...

    Thinking mode prefix: <｜begin▁of▁sentence｜>{system prompt}<｜User｜>{query}<｜Assistant｜><think>
    """
    if is_thinking:
        # Thinking mode uses <think> token
        prefix = (
            f"<｜begin▁of▁sentence｜>{system_prompt}<｜User｜>{user_query}<｜Assistant｜><think>"
        )
    else:
        # Non-thinking mode uses </think> closing token
        prefix = (
            f"<｜begin▁of▁sentence｜>{system_prompt}<｜User｜>{user_query}<｜Assistant｜></think>"
        )
    
    if context:
        # Append context to prefix
        return context + prefix
    return prefix


def format_deepseek_v31_tool_call_prompt(system_prompt: str, tool_description: str, user_query: str) -> str:
    """
    Format tool call prompt according to DeepSeek V3.1 requirements.

    Toolcall format: <｜begin▁of▁sentence｜>{system prompt}\n\n{tool_description}<｜User｜>{query}<｜Assistant｜></think>
    """
    return (
        f"<｜begin▁of▁sentence｜>{system_prompt}\n\n{tool_description}"
        f"<｜User｜>{user_query}<｜Assistant｜></think>"
    )



def parse_deepseek_v31_tool_markup(text: str) -> tuple[str, list[Dict[str, Any]]]:
    """Parse DeepSeek V3.1 tool-call markup into structured tool_use blocks.
    
    Recognizes sections of the form:
      <｜tool▁calls▁begin｜><｜tool▁call▁begin｜>tool_name<｜tool▁sep｜>{"key": "value"}<｜tool▁call▁end｜><｜tool▁calls▁end｜>
    
    Returns (remaining_text_without_markup, tool_use_blocks)
    """
    if not text:
        return text, []
    
    # Normalize fullwidth characters and special glyphs
    try:
        normalized_text = text.replace("｜", "|").replace("▁", "_")
    except Exception:
        normalized_text = text
    
    BEG = "<|tool_calls_begin|>"
    END = "<|tool_calls_end|>"
    CALL_BEG = "<|tool_call_begin|>"
    CALL_END = "<|tool_call_end|>"
    SEP = "<|tool_sep|>"
    
    out_text_parts: list[str] = []
    calls: list[Dict[str, Any]] = []
    pos = 0
    n = len(normalized_text)
    
    while pos < n:
        i = normalized_text.find(BEG, pos)
        if i == -1:
            out_text_parts.append(normalized_text[pos:])
            break
        # text before the section
        out_text_parts.append(normalized_text[pos:i])
        j = normalized_text.find(END, i)
        if j == -1:
            # No closing tag; keep as plain text
            out_text_parts.append(normalized_text[i:])
            break
        
        section = normalized_text[i + len(BEG):j]
        call_pos = 0
        call_n = len(section)
        
        while call_pos < call_n:
            call_i = section.find(CALL_BEG, call_pos)
            if call_i == -1:
                break
            call_j = section.find(CALL_END, call_i)
            if call_j == -1:
                break
            
            call_content = section[call_i + len(CALL_BEG):call_j]
            sep_idx = call_content.find(SEP)
            if sep_idx != -1:
                name = call_content[:sep_idx].strip()
                args_raw = call_content[sep_idx + len(SEP):].strip()
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
            call_pos = call_j + len(CALL_END)
        
        # Skip the whole section in output text
        pos = j + len(END)
    
    return "".join(out_text_parts), calls


_LONGCAT_TOOL_CALL_RE = re.compile(r"<longcat_tool_call>(.*?)</longcat_tool_call>", re.DOTALL)


def parse_longcat_tool_markup(text: str) -> tuple[str, list[Dict[str, Any]]]:
    if not text or "<longcat_tool_call" not in text:
        return text, []

    calls: list[Dict[str, Any]] = []
    out_parts: list[str] = []
    last_idx = 0
    for match in _LONGCAT_TOOL_CALL_RE.finditer(text):
        start, end = match.span()
        out_parts.append(text[last_idx:start])
        payload_raw = (match.group(1) or "").strip()
        try:
            payload = json_loads_safe(payload_raw)
        except Exception:
            payload = {}
        if isinstance(payload, dict):
            name = payload.get("name") or ""
            arguments = payload.get("arguments")
            if not isinstance(arguments, dict):
                arguments = {}
            calls.append(
                {
                    "type": "tool_use",
                    "id": f"call_{uuid.uuid4().hex}",
                    "name": name,
                    "input": arguments,
                }
            )
        last_idx = end
    if not calls:
        return text, []
    out_parts.append(text[last_idx:])
    return "".join(out_parts), calls


def normalize_longcat_completion(text: Optional[str], model_name: Optional[str] = None) -> str:
    if not text:
        return ""
    out = str(text)
    if "</longcat_s>" in out:
        out = out.replace("</longcat_s>", " ")
    if "[Round" in out and "ASSISTANT:" in out:
        idx = out.rfind("ASSISTANT:")
        if idx != -1:
            out = out[idx + len("ASSISTANT:") :]
    return out.strip()


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
    thinking_blocks: List[Dict[str, Any]] = []
    model_name = str(requested_model or oai.get("model") or "")
    is_deepseek_model = "deepseek" in model_name.lower()

    # Preserve declared tool names (if any) so we can map upstream aliases back to canonical casing
    declared_tool_names: list[str] = []
    declared_tool_lookup: Dict[str, str] = {}
    try:
        for t in tools or []:
            if not isinstance(t, dict):
                continue
            name: Optional[str] = None
            fn = t.get("function")
            if isinstance(fn, dict):
                maybe = fn.get("name")
                if isinstance(maybe, str) and maybe.strip():
                    name = maybe.strip()
            if not name:
                maybe = t.get("name")
                if isinstance(maybe, str) and maybe.strip():
                    name = maybe.strip()
            if name:
                declared_tool_names.append(name)
    except Exception:
        declared_tool_names = []
    if declared_tool_names:
        try:
            declared_tool_lookup = {n.lower(): n for n in declared_tool_names}
        except Exception:
            declared_tool_lookup = {}
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
        def _map_name(n: Optional[str]) -> str:
            try:
                raw = (n or "").strip()
                if not raw:
                    return ""
                if raw in declared_tool_names:
                    return raw
                low = raw.lower()
                if declared_tool_lookup and low in declared_tool_lookup:
                    return declared_tool_lookup[low]
                mapped = settings.map_tool_name(requested_model or oai.get("model"), raw)
                out = (mapped or raw) if isinstance(mapped, str) else raw
                # Common synonyms (case-insensitive)
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
                    # Additional synonyms for better compatibility
                    "run_terminal_cmd": "Bash",
                    "glob_file_search": "Glob",
                    "todo_write": "TodoWrite",
                }
                out = synonyms.get(out) or synonyms.get(out.lower()) or out
                if out in declared_tool_names:
                    return out
                if declared_tool_lookup:
                    low_out = out.lower()
                    if low_out in declared_tool_lookup:
                        return declared_tool_lookup[low_out]
                if not declared_tool_names and out and (any(ch.isupper() for ch in out) and "_" not in out):
                    import re as _re
                    s1 = _re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", out)
                    s2 = _re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s1)
                    out = s2.lower()
                return out
            except Exception:
                return n or ""

        first_choice = (oai.get("choices") or [{}])[0]
        message = first_choice.get("message") or {}
        text = message.get("content") or ""
        if isinstance(text, list):
            try:
                text = "".join(str(part.get("text", "")) for part in text if isinstance(part, dict))
            except Exception:
                text = ""
        # If available, include reasoning content as a dedicated thinking block (for proper UI display)
        reasoning = message.get("reasoning_content")
        if reasoning:
            # DeepSeek sometimes returns arrays; collapse to plain string
            if isinstance(reasoning, (list, tuple)):
                joined = "".join(str(part) for part in reasoning if part is not None)
            else:
                joined = str(reasoning)
            if joined.strip():
                thinking_blocks.append({"type": "thinking", "thinking": joined})
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
                    "name": _map_name((tc.get("function") or {}).get("name")),
                    "input": args,
                }
            )
        # Parse Cloud Code MCP tool-call markup first
        if text:
            text, longcat_calls = parse_longcat_tool_markup(text)
            for c in longcat_calls or []:
                try:
                    c["name"] = _map_name(c.get("name"))
                except Exception:
                    ...
            tool_calls_block.extend(longcat_calls)
            text, markup_calls = parse_mcp_tool_markup(text)
            # apply tool name mapping
            for c in markup_calls or []:
                try:
                    c["name"] = _map_name(c.get("name"))
                except Exception:
                    ...
            tool_calls_block.extend(markup_calls)
            # Parse DeepSeek V3.1 tool-call markup
            text, deepseek_calls = parse_deepseek_v31_tool_markup(text)
            for c in deepseek_calls or []:
                try:
                    c["name"] = _map_name(c.get("name"))
                except Exception:
                    ...
            tool_calls_block.extend(deepseek_calls)
            # DeepSeek tool calls often prepend textual "Thinking…" placeholders. Promote them to thinking blocks.
            stripped = text.strip()
            if stripped and is_deepseek_model and tool_calls_block:
                normalized_prefix = stripped[:64].lower().replace("…", "...")
                normalized_prefix = normalized_prefix.replace("：", ":")
                normalized_full = stripped.lower().replace("…", "...").strip()
                placeholder_values = {"thinking...", "thinking.", "thinking", "thought...", "analysis...", "analysis."}
                if normalized_prefix.startswith("thinking") or normalized_prefix.startswith("thought") or normalized_prefix.startswith("analysis"):
                    if normalized_full not in placeholder_values:
                        thinking_blocks.append({"type": "thinking", "thinking": stripped})
                    stripped = ""
                if not stripped:
                    text = ""
                else:
                    text = stripped
        # Fallback: use sglang parser on textual content when no structured tool_calls
        if not tcalls and text:
            text, parsed_calls = _maybe_parse_tools_with_sglang(text)
            for c in parsed_calls or []:
                try:
                    c["name"] = _map_name(c.get("name"))
                except Exception:
                    ...
            tool_calls_block.extend(parsed_calls)
    except Exception:
        text = ""

    text = normalize_longcat_completion(text, requested_model or oai.get("model"))

    finish_reason = None
    try:
        fr = (oai.get("choices") or [{}])[0].get("finish_reason")
    except Exception:
        fr = None

    anthropic_stop_reason = map_finish_reason(fr)

    usage = oai.get("usage") or {}

    content_blocks: List[Dict[str, Any]] = []
    if text:
        trimmed_text = str(text).strip()
        if trimmed_text:
            content_blocks.append({"type": "text", "text": trimmed_text})
    if thinking_blocks:
        content_blocks = thinking_blocks + content_blocks
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
    if not text_chunk:
        return "", []

    cleaned_chunk, markup_calls = parse_longcat_tool_markup(text_chunk)
    cleaned_chunk = normalize_longcat_completion(cleaned_chunk, model_name)
    if not tools:
        return cleaned_chunk, markup_calls
    if markup_calls:
        return cleaned_chunk, markup_calls
    text_chunk = cleaned_chunk

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
        normal_text = normalize_longcat_completion(normal_text, model_name)
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
        if isinstance(s, str):
            try:
                repaired = _repair_invalid_json_escapes(s)
                if repaired != s:
                    import json

                    return json.loads(repaired)
            except Exception:
                ...
        return {}


_INVALID_ESCAPE_RE = re.compile(r"\\(?![\\\"/bfnrtu])")


def _repair_invalid_json_escapes(raw: str) -> str:
    """Best-effort fix for JSON strings containing invalid escape sequences."""
    try:
        return _INVALID_ESCAPE_RE.sub("", raw)
    except Exception:
        return raw
