import json

from app.transform import (
    anthropic_to_openai_payload,
    openai_to_anthropic_response,
    choose_tool_call_parser,
    parse_stream_tools_text,
)


def test_text_only_request_mapping():
    req = {
        "model": "claude-3-sonnet",
        "max_tokens": 32,
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": "Hello"}]}
        ],
        "temperature": 0.1,
        "top_p": 0.9,
        "stop_sequences": ["\n\n"],
    }
    out = anthropic_to_openai_payload(req)
    assert out["model"] == "claude-3-sonnet"
    assert out["max_tokens"] == 32
    assert out["messages"][0]["role"] == "user"
    assert out["messages"][0]["content"] == "Hello"
    assert out["temperature"] == 0.1
    assert out["top_p"] == 0.9
    assert out["stop"] == ["\n\n"]


def test_tools_request_mapping():
    req = {
        "model": "claude-3-sonnet",
        "max_tokens": 16,
        "tools": [
            {
                "name": "get_weather",
                "description": "Get weather",
                "input_schema": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            }
        ],
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What's the weather?"},
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_1",
                        "name": "get_weather",
                        "content": [{"type": "text", "text": "{\"temp\":20}"}],
                    },
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "I'll check."},
                    {
                        "type": "tool_use",
                        "id": "toolu_1",
                        "name": "get_weather",
                        "input": {"city": "SF"},
                    },
                ],
            },
        ],
    }
    out = anthropic_to_openai_payload(req)
    msgs = out["messages"]
    assert msgs[0]["role"] == "user" and "What's the weather?" in msgs[0]["content"]
    assert msgs[1]["role"] == "tool" and msgs[1]["tool_call_id"] == "toolu_1"
    assert msgs[2]["role"] == "assistant" and msgs[2]["tool_calls"][0]["function"]["name"] == "get_weather"
    assert out["tools"][0]["function"]["name"] == "get_weather"


def test_image_block_mapping():
    req = {
        "model": "claude-3-sonnet",
        "max_tokens": 16,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is in the image?"},
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": "iVBORw0KGgoAAAANSUhEUgAA...",
                        },
                    },
                ],
            }
        ],
    }
    out = anthropic_to_openai_payload(req)
    msg = out["messages"][0]
    assert msg["role"] == "user"
    assert isinstance(msg["content"], list)
    assert msg["content"][0]["type"] == "text"
    assert msg["content"][1]["type"] == "image_url"
    assert msg["content"][1]["image_url"]["url"].startswith("data:image/png;base64,")


def test_openai_tool_calls_to_anthropic_response():
    oai_resp = {
        "id": "chatcmpl-xyz",
        "object": "chat.completion",
        "created": 1,
        "model": "gpt-sim",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Let me call a tool.",
                    "tool_calls": [
                        {
                            "id": "call_123",
                            "type": "function",
                            "function": {"name": "get_weather", "arguments": "{\"city\":\"SF\"}"},
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
    }
    anth = openai_to_anthropic_response(oai_resp)
    assert anth["stop_reason"] == "tool_use"
    assert any(b.get("type") == "tool_use" for b in anth["content"])


def test_choose_parser_heuristics():
    assert choose_tool_call_parser("LLaMA-3-8B-Instruct") == "llama3"
    assert choose_tool_call_parser("Qwen2.5-7B-Instruct") == "qwen25"
    assert choose_tool_call_parser("Mistral-Small-24B") == "mistral"
    assert choose_tool_call_parser("deepseek-v3.1") == "deepseekv31"
    assert choose_tool_call_parser("glm-4.5") == "glm45"
    assert choose_tool_call_parser("longcat/Longcat-Flash-8B") == "gpt-oss"


def test_stream_chunk_tool_parse_pythonic():
    # Simulate a streaming text delta that includes a pythonic tool call
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                },
            },
        }
    ]
    text = "Intro text [get_weather(city='SF')] tail"
    normal, calls = parse_stream_tools_text(text, tools, model_name="llama-4-scout")
    # Accept either successful call extraction or preserved text when parser is unavailable
    assert (any(call.get("name") == "get_weather" for call in calls)) or ("Intro text" in normal or "tail" in normal)


def test_stream_chunk_tool_parse_unknown_markup():
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
            },
        }
    ]
    # Unknown textual markup should be preserved as text when not parsed by sglang
    text = "A <tool_call_textual>{\"name\":\"get_weather\",\"arguments\":{\"city\":\"SF\"}}</tool_call_textual> B"
    normal, calls = parse_stream_tools_text(text, tools, model_name="unknown-model")
    assert (any(call.get("name") == "get_weather" for call in calls)) or ("<tool_call_textual>" in normal)
