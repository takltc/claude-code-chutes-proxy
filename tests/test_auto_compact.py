import pytest

from app.auto_compact import AutoCompactError, ensure_context_within_limits
from app.schemas.anthropic import MessageInput


def _msg(role: str, text: str) -> MessageInput:
    return MessageInput(role=role, content=[{"type": "text", "text": text}])


@pytest.mark.asyncio
async def test_ensure_context_noop_when_under_threshold():
    messages = [
        _msg("user", "short"),
        _msg("assistant", "ack"),
    ]

    compacted, stats = await ensure_context_within_limits(
        model="claude-3-5-sonnet-20240620",
        system="",
        messages=messages,
        client=None,
        headers={},
        context_window=2000,
        reserve_tokens=256,
        buffer_ratio=0.9,
        tail_reserve=6,
        summary_model=None,
        summary_max_tokens=512,
        summary_keep_last=4,
        auto_condense_percent=95,
        summary_provider=lambda *args, **kwargs: (_async_fail()),
    )

    assert [m.model_dump() for m in compacted] == [m.model_dump() for m in messages]
    assert stats.summary_added is False
    assert stats.truncated is False
    assert stats.after_tokens <= stats.threshold


async def _async_fail():
    raise AssertionError("summary provider should not be called")


@pytest.mark.asyncio
async def test_ensure_context_uses_summary_when_exceeding_threshold():
    long_text = "Repeat this instruction carefully. " * 200
    messages = [
        _msg("user", "initial"),
        _msg("assistant", "ack"),
        _msg("user", long_text),
        _msg("assistant", "processing"),
        _msg("user", "follow up"),
    ]

    async def fake_summary(client, headers, model, prompt_text, max_tokens):
        assert "Conversation History" in prompt_text
        return "Summary: condensed", 120

    compacted, stats = await ensure_context_within_limits(
        model="claude-3-5-sonnet-20240620",
        system="",
        messages=messages,
        client=None,
        headers={},
        context_window=1000,
        reserve_tokens=0,
        buffer_ratio=0.6,
        tail_reserve=6,
        summary_model=None,
        summary_max_tokens=256,
        summary_keep_last=1,
        auto_condense_percent=50,
        summary_provider=fake_summary,
    )

    assert stats.summary_added is True
    assert any("Summary:" in block.get("text", "") for block in compacted[1].content if isinstance(block, dict))
    assert any("Summary:" in block.get("text", "") for block in compacted[1].content if isinstance(block, dict))
    assert len(compacted) <= len(messages)


@pytest.mark.asyncio
async def test_ensure_context_fallback_to_sliding_window_when_summary_fails():
    long_text = "Repeat this instruction carefully. " * 200
    messages = [
        _msg("user", "initial"),
        _msg("assistant", "ack"),
        _msg("user", long_text),
        _msg("assistant", "processing"),
        _msg("user", "follow up"),
    ]

    async def failing_summary(*args, **kwargs):
        raise AutoCompactError("summary failed")

    compacted, stats = await ensure_context_within_limits(
        model="claude-3-5-sonnet-20240620",
        system="",
        messages=messages,
        client=None,
        headers={},
        context_window=1000,
        reserve_tokens=0,
        buffer_ratio=0.6,
        tail_reserve=6,
        summary_model=None,
        summary_max_tokens=256,
        summary_keep_last=2,
        auto_condense_percent=50,
        summary_provider=failing_summary,
    )

    assert stats.summary_added is False
    assert stats.truncated is True
    assert len(compacted) < len(messages)


@pytest.mark.asyncio
async def test_threshold_respects_safety_tokens():
    long_text = "word " * 2200
    messages = [
        _msg("user", "intro"),
        _msg("assistant", "ok"),
        _msg("user", long_text),
        _msg("assistant", "noted"),
    ]

    async def fake_summary(*_args, **_kwargs):
        return "condensed", 64

    compacted, stats = await ensure_context_within_limits(
        model="claude-3-5-sonnet-20240620",
        system="",
        messages=messages,
        client=None,
        headers={},
        context_window=2000,
        reserve_tokens=0,
        buffer_ratio=0.95,
        tail_reserve=2,
        summary_model=None,
        summary_max_tokens=128,
        summary_keep_last=1,
        auto_condense_percent=50,
        safety_tokens=400,
        summary_provider=fake_summary,
    )

    assert stats.summary_added is True
    assert stats.after_tokens <= stats.threshold
    assert stats.threshold == 1600


@pytest.mark.asyncio
async def test_ensure_context_raises_when_unable_to_fit():
    gigantic = "A" * 8000
    messages = [
        _msg("user", "start"),
        _msg("assistant", "ok"),
        _msg("user", gigantic),
    ]

    async def empty_summary(*args, **kwargs):
        return "", 0

    with pytest.raises(AutoCompactError):
        await ensure_context_within_limits(
            model="claude-3-5-sonnet-20240620",
            system="",
            messages=messages,
            client=None,
            headers={},
            context_window=1000,
            reserve_tokens=500,
            buffer_ratio=0.8,
            tail_reserve=6,
            summary_model=None,
            summary_max_tokens=128,
            summary_keep_last=2,
            auto_condense_percent=70,
            summary_provider=empty_summary,
        )


def test_sliding_window_truncate_reduces_length():
    from app.auto_compact import sliding_window_truncate

    messages = [
        _msg('user', 'm0'),
        _msg('assistant', 'm1'),
        _msg('user', 'm2'),
    ]

    compacted = sliding_window_truncate(messages)
    assert len(compacted) < len(messages)
    assert compacted[0].role == 'user'
    assert compacted[-1].role == 'user'
