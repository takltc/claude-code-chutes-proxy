from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple

import httpx

from .schemas.anthropic import MessageInput
from .config import settings

try:  # pragma: no cover
    import tiktoken
except Exception:  # pragma: no cover
    tiktoken = None  # type: ignore


SUMMARY_PROMPT = """\
Your task is to create a detailed summary of the conversation so far, paying close attention to the user's explicit requests and your previous actions.\nThis summary should be thorough in capturing technical details, code patterns, and architectural decisions that would be essential for continuing with the conversation and supporting any continuing tasks.\n\nStructure the summary as:\nContext:\n  1. Previous Conversation\n  2. Current Work\n  3. Key Technical Concepts\n  4. Relevant Files and Code\n  5. Problem Solving\n  6. Pending Tasks and Next Steps\n\nOutput only the summary text."""

TOKEN_BUFFER_PERCENTAGE = 0.1
N_MESSAGES_TO_KEEP = 4


class AutoCompactError(RuntimeError):
    ...


@dataclass
class ContextStats:
    before_tokens: int
    after_tokens: int
    threshold: int
    removed_messages: int
    summary_added: bool
    truncated: bool
    summary_tokens: int = 0


class TokenEstimator:
    def __init__(self, model: Optional[str]) -> None:
        self.encoding = None
        if tiktoken is None:
            return
        try:
            if model:
                self.encoding = tiktoken.encoding_for_model(model)
        except Exception:
            self.encoding = None
        if self.encoding is None:
            try:
                self.encoding = tiktoken.get_encoding("cl100k_base")
            except Exception:
                self.encoding = None

    def count_text(self, text: str) -> int:
        if not text:
            return 0
        if self.encoding is not None:
            try:
                return len(self.encoding.encode(text))
            except Exception:
                ...
        return max(1, len(text) // 4)

    def count_message(self, message: Dict[str, Any]) -> int:
        content = message.get("content")
        text = convert_content_to_text(content)
        return self.count_text(text) + 3


def convert_content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    parts: List[str] = []
    for block in content or []:
        btype = block.get("type") if isinstance(block, dict) else getattr(block, "type", None)
        if btype == "text":
            text = block.get("text") if isinstance(block, dict) else getattr(block, "text", "")
            if text:
                parts.append(str(text))
        elif btype in ("tool_use", "tool_result"):
            try:
                snippet = json.dumps(block, ensure_ascii=False)
            except Exception:
                snippet = str(block)
            parts.append(snippet)
        else:
            try:
                parts.append(json.dumps(block, ensure_ascii=False))
            except Exception:
                parts.append(str(block))
    return "\n".join(parts)


SummaryProvider = Callable[[httpx.AsyncClient, Dict[str, str], str, str, int], Awaitable[Tuple[str, int]]]


async def ensure_context_within_limits(
    model: str,
    system: Any,
    messages: List[MessageInput],
    *,
    client: Optional[httpx.AsyncClient],
    headers: Dict[str, str],
    context_window: int,
    reserve_tokens: int,
    buffer_ratio: float,
    tail_reserve: int,
    summary_model: Optional[str],
    summary_max_tokens: int,
    summary_keep_last: int,
    auto_condense_percent: int,
    summary_provider: Optional[SummaryProvider] = None,
) -> Tuple[List[MessageInput], ContextStats]:
    if context_window <= 0:
        raise AutoCompactError("Invalid context window")

    estimator = TokenEstimator(model)
    system_text = convert_content_to_text(system)
    system_tokens = estimator.count_text(system_text)

    raw_messages = [m.model_dump() if isinstance(m, MessageInput) else dict(m) for m in messages]
    per_tokens = [estimator.count_message(m) for m in raw_messages]
    before_total = system_tokens + sum(per_tokens)

    allowed_tokens = int(context_window * (1 - TOKEN_BUFFER_PERCENTAGE)) - reserve_tokens
    allowed_tokens = max(512, allowed_tokens)
    threshold_tokens = max(int(context_window * buffer_ratio) - reserve_tokens, allowed_tokens)

    if before_total <= threshold_tokens:
        return messages, ContextStats(
            before_tokens=before_total,
            after_tokens=before_total,
            threshold=threshold_tokens,
            removed_messages=0,
            summary_added=False,
            truncated=False,
        )

    if summary_provider is None:
        summary_provider = summarize_via_api
    if client is None and summary_provider is summarize_via_api:
        raise AutoCompactError("HTTP client is required for summarization")

    context_percent = (100 * before_total) / max(1, context_window)

    summary_added = False
    summary_tokens = 0
    rebuilt: Optional[List[MessageInput]] = None

    if context_percent >= auto_condense_percent or before_total > allowed_tokens:
        try:
            summary_text, summary_tokens = await summary_provider(
                client,
                headers,
                summary_model or model,
                build_summary_prompt(raw_messages, tail_reserve),
                summary_max_tokens,
            )
            summary_text = summary_text.strip()
        except Exception:
            summary_text = ""

        if summary_text:
            summary_added = True
            summary_message = MessageInput.model_validate(
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": summary_text}],
                }
            )
            keep_tail = max(1, min(summary_keep_last, len(messages)))
            tail_slice = messages[-keep_tail:]
            rebuilt = [messages[0]]
            if messages[0] in tail_slice:
                tail_slice = tail_slice[1:]
            rebuilt.append(summary_message)
            rebuilt.extend(tail_slice)
            per_tokens = [estimator.count_message(m.model_dump()) for m in rebuilt]
            after_total = system_tokens + sum(per_tokens)
            if after_total <= threshold_tokens:
                return (
                    rebuilt,
                    ContextStats(
                        before_tokens=before_total,
                        after_tokens=after_total,
                        threshold=threshold_tokens,
                        removed_messages=len(messages) - len(rebuilt),
                        summary_added=True,
                        truncated=False,
                        summary_tokens=summary_tokens,
                    ),
                )

    truncated_messages = rebuilt if rebuilt is not None else messages
    while len(truncated_messages) > 2:
        truncated_messages = sliding_window_truncate(truncated_messages)
        per_tokens = [
            estimator.count_message(m.model_dump() if isinstance(m, MessageInput) else dict(m))
            for m in truncated_messages
        ]
        after_total = system_tokens + sum(per_tokens)
        if after_total <= threshold_tokens:
            return (
                truncated_messages,
                ContextStats(
                    before_tokens=before_total,
                    after_tokens=after_total,
                    threshold=threshold_tokens,
                    removed_messages=len(messages) - len(truncated_messages),
                    summary_added=summary_added,
                    truncated=True,
                    summary_tokens=summary_tokens,
                ),
            )

    raise AutoCompactError("Context still exceeds maximum token allowance after summarization")


def build_summary_prompt(messages: List[Dict[str, Any]], tail_reserve: int) -> str:
    keep_tail = max(1, min(tail_reserve, len(messages)))
    target = messages[:-keep_tail] if keep_tail < len(messages) else messages
    if len(target) <= 1:
        target = messages
    lines: List[str] = []
    for msg in target:
        role = msg.get("role", "assistant")
        text = convert_content_to_text(msg.get("content"))
        if text:
            lines.append(f"{role.upper()}: {text}")
    conversation = "\n".join(lines)
    return f"{SUMMARY_PROMPT}\n\nConversation History:\n{conversation}"


def sliding_window_truncate(messages: List[MessageInput]) -> List[MessageInput]:
    if len(messages) <= 2:
        return messages
    first = messages[0]
    rest = messages[1:]
    remove_count = max(1, len(rest) // 2)
    if remove_count % 2:
        remove_count += 1
    if remove_count >= len(rest):
        remove_count = max(1, len(rest) - 1)
    truncated = [first]
    truncated.extend(rest[remove_count:])
    return truncated


async def summarize_via_api(
    client: httpx.AsyncClient,
    headers: Dict[str, str],
    summary_model: str,
    prompt_text: str,
    max_tokens: int,
) -> Tuple[str, int]:
    url = f"{settings.chutes_base_url.rstrip('/')}/v1/chat/completions"
    payload = {
        "model": summary_model,
        "messages": [
            {"role": "system", "content": SUMMARY_PROMPT},
            {"role": "user", "content": prompt_text},
        ],
        "max_tokens": max_tokens,
        "stream": False,
    }
    headers_summary = {k: v for k, v in headers.items() if k.lower() != "accept"}
    resp = await client.post(
        url,
        json=payload,
        headers=headers_summary,
        timeout=httpx.Timeout(60.0),
    )
    resp.raise_for_status()
    data = resp.json()
    choices = data.get("choices") or []
    if not choices:
        raise AutoCompactError("Summarization returned no choices")
    content = choices[0].get("message", {}).get("content")
    if isinstance(content, list):
        summary_text = "".join(
            block.get("text", "") for block in content if isinstance(block, dict)
        )
    else:
        summary_text = str(content or "")
    usage = data.get("usage") or {}
    completion_tokens = int(usage.get("completion_tokens") or 0)
    if not summary_text.strip():
        raise AutoCompactError("Summarization produced empty text")
    return summary_text.strip(), completion_tokens
