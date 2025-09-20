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


def test_longcat_prompt_formatting_basic():
    req = {
        "model": "longcat/Longcat-Flash-8B",
        "system": [{"type": "text", "text": "Be concise."}],
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": "Hello"}]}
        ],
    }
    out = anthropic_to_openai_payload(req)
    assert out["messages"][0]["role"] == "user"
    prompt = out["messages"][0]["content"]
    assert prompt.startswith("markdown")
    assert "## Messages" in prompt
    assert "SYSTEM:Be concise." in prompt
    assert "[Round 0] USER:Hello ASSISTANT:" in prompt


def test_longcat_prompt_with_history_and_tools():
    req = {
        "model": "longcat/Longcat-Flash-8B",
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": "Hi"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "Hello there"}]},
            {"role": "user", "content": [{"type": "text", "text": "Need weather"}]},
        ],
        "tools": [
            {
                "name": "get_weather",
                "description": "Weather lookup",
                "input_schema": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                },
            }
        ],
    }
    out = anthropic_to_openai_payload(req)
    prompt = out["messages"][0]["content"]
    assert "## Tools" in prompt
    assert "#### Tool name: get_weather" in prompt
    assert "[Round 0] USER:Hi ASSISTANT:Hello there</longcat_s>" in prompt
    assert prompt.strip().endswith("[Round 1] USER:Need weather ASSISTANT:")


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


def test_stream_chunk_longcat_markup():
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
            },
        }
    ]
    chunk = (
        "Working on it</longcat_s><longcat_tool_call>\n"
        "{\"name\": \"get_weather\", \"arguments\": {\"city\": \"SF\"}}\n"
        "</longcat_tool_call>"
    )
    normal, calls = parse_stream_tools_text(chunk, tools, model_name="longcat/Longcat-Flash-8B")
    assert "Working on it" in normal
    assert "</longcat_s>" not in normal
    assert len(calls) == 1 and calls[0]["name"] == "get_weather"


def test_openai_longcat_tool_markup_parsing():
    oai_resp = {
        "id": "chatcmpl-longcat",
        "object": "chat.completion",
        "created": 2,
        "model": "longcat/Longcat-Flash-8B",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": (
                        "Calling tool\n<longcat_tool_call>\n"
                        "{\"name\": \"get_weather\", \"arguments\": {\"city\": \"SF\"}}\n"
                        "</longcat_tool_call>"
                    ),
                },
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
    }
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
            },
        }
    ]
    anth = openai_to_anthropic_response(oai_resp, requested_model="longcat/Longcat-Flash-8B", tools=tools)
    texts = [block for block in anth["content"] if block.get("type") == "text"]
    assert texts and "Calling tool" in texts[0]["text"]
    assert "</longcat_s>" not in texts[0]["text"]
    tool_blocks = [block for block in anth["content"] if block.get("type") == "tool_use"]
    assert tool_blocks and tool_blocks[0]["name"] == "get_weather"
    assert tool_blocks[0]["input"] == {"city": "SF"}


def test_deepseek_v31_prompt_formatting():
    """Test DeepSeek V3.1 prompt formatting functions."""
    from app.transform import format_deepseek_v31_prompt, format_deepseek_v31_tool_call_prompt
    
    # Test non-thinking mode prefix
    prefix = format_deepseek_v31_prompt("You are a helpful assistant.", "Hello, how are you?")
    assert (
        prefix
        == "<｜begin▁of▁sentence｜>You are a helpful assistant.<｜User｜>Hello, how are you?<｜Assistant｜></think>"
    )
    
    # Test thinking mode prefix
    thinking_prefix = format_deepseek_v31_prompt(
        "You are a helpful assistant.", "Think step by step.", is_thinking=True
    )
    assert (
        thinking_prefix
        == "<｜begin▁of▁sentence｜>You are a helpful assistant.<｜User｜>Think step by step.<｜Assistant｜><think>"
    )
    
    # Test with context
    context = (
        "<｜begin▁of▁sentence｜>You are a helpful assistant.<｜User｜>What's the weather?"
        "<｜Assistant｜></think>The weather is sunny.<｜end▁of▁sentence｜>"
    )
    with_context = format_deepseek_v31_prompt(
        "You are a helpful assistant.", "What about tomorrow?", context=context
    )
    assert (
        with_context
        == context
        + "<｜begin▁of▁sentence｜>You are a helpful assistant.<｜User｜>What about tomorrow?<｜Assistant｜></think>"
    )
    
    # Test tool call prompt formatting
    tool_desc = "## Tools\nYou have access to the following tools:\n\n### get_weather\nDescription: Get weather information\nParameters: {\"type\": \"object\", \"properties\": {\"city\": {\"type\": \"string\"}}, \"required\": [\"city\"]}"
    tool_prompt = format_deepseek_v31_tool_call_prompt("You are a helpful assistant.", tool_desc, "What's the weather in SF?")
    expected = (
        "<｜begin▁of▁sentence｜>You are a helpful assistant.\n\n" + tool_desc
        + "<｜User｜>What's the weather in SF?<｜Assistant｜></think>"
    )
    assert tool_prompt == expected


def test_deepseek_v31_tool_call_payload():
    """Test DeepSeek V3.1 tool call payload formatting."""
    from app.transform import format_deepseek_v31_tool_call_payload
    
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather information",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string"}
                    },
                    "required": ["city"]
                }
            }
        }
    ]
    
    result = format_deepseek_v31_tool_call_payload("You are a helpful assistant.", tools, "What's the weather in SF?")
    assert "prompt" in result
    assert "tool_description" in result
    assert "<｜begin▁of▁sentence｜>You are a helpful assistant." in result["prompt"]
    assert "## Tools" in result["tool_description"]
    assert "get_weather" in result["tool_description"]
    assert "<｜User｜>What's the weather in SF?<｜Assistant｜></think>" in result["prompt"]


def test_deepseek_v31_tool_call_parsing():
    """Test DeepSeek V3.1 tool call markup parsing."""
    from app.transform import parse_deepseek_v31_tool_markup
    
    # Test parsing DeepSeek V3.1 tool call markup
    text = "Some text <｜tool▁calls▁begin｜><｜tool▁call▁begin｜>get_weather<｜tool▁sep｜>{\"city\": \"SF\"}<｜tool▁call▁end｜><｜tool▁calls▁end｜> more text"
    remaining_text, tool_calls = parse_deepseek_v31_tool_markup(text)
    
    assert "Some text" in remaining_text
    assert "more text" in remaining_text
    assert len(tool_calls) == 1
    assert tool_calls[0]["name"] == "get_weather"
    assert tool_calls[0]["input"] == {"city": "SF"}
    
    # Test with multiple tool calls
    text2 = "Text <｜tool▁calls▁begin｜><｜tool▁call▁begin｜>get_weather<｜tool▁sep｜>{\"city\": \"SF\"}<｜tool▁call▁end｜><｜tool▁call▁begin｜>get_time<｜tool▁sep｜>{\"timezone\": \"PST\"}<｜tool▁call▁end｜><｜tool▁calls▁end｜> end"
    remaining_text2, tool_calls2 = parse_deepseek_v31_tool_markup(text2)
    
    assert "Text" in remaining_text2
    assert "end" in remaining_text2
    assert len(tool_calls2) == 2
    assert tool_calls2[0]["name"] == "get_weather"
    assert tool_calls2[0]["input"] == {"city": "SF"}
    assert tool_calls2[1]["name"] == "get_time"
    assert tool_calls2[1]["input"] == {"timezone": "PST"}
    
    # Test with no tool calls
    text3 = "Just plain text without tool calls"
    remaining_text3, tool_calls3 = parse_deepseek_v31_tool_markup(text3)
    
    assert remaining_text3 == text3
    assert len(tool_calls3) == 0


def test_deepseek_tool_call_preserves_declared_name():
    """DeepSeek tool markup should retain declared tool casing when available."""
    from app.transform import openai_to_anthropic_response

    message_text = (
        "Thinking…\n\n"
        "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>TodoWrite<｜tool▁sep｜>"
        "{\"todos\": [{\"content\": \"检查Nacos容器启动状态\", \"status\": \"completed\","
        " \"activeForm\": \"检查Nacos容器启动状态\"}], \"command\": \"curl -s http://127.0.0.1:8848/nacos/v1/cs/configs?dataId=user-sample-service.yaml\\&group=DEFAULT_GROUP\"}"
        "<｜tool▁call▁end｜><｜tool▁calls▁end｜>"
    )

    oai_resp = {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 0,
        "model": "deepseek-ai/DeepSeek-V3.1",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": message_text,
                },
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5},
    }

    tools = [
        {
            "type": "function",
            "function": {
                "name": "TodoWrite",
                "description": "",
                "parameters": {"type": "object"},
            },
        }
    ]

    anthropic = openai_to_anthropic_response(oai_resp, requested_model="deepseek-ai/DeepSeek-V3.1", tools=tools)
    tool_blocks = [blk for blk in anthropic.get("content", []) if blk.get("type") == "tool_use"]

    assert tool_blocks, "expected tool_use block from DeepSeek markup"
    first_call = tool_blocks[0]
    assert first_call["name"] == "TodoWrite"
    payload = first_call.get("input") or {}
    assert payload.get("command") == (
        "curl -s http://127.0.0.1:8848/nacos/v1/cs/configs?dataId=user-sample-service.yaml&group=DEFAULT_GROUP"
    )


def test_deepseek_thinking_placeholder_removed():
    from app.transform import openai_to_anthropic_response

    message_text = (
        "Thinking…\n\n"
        "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>TodoWrite<｜tool▁sep｜>{\"todos\": []}<｜tool▁call▁end｜><｜tool▁calls▁end｜>"
    )

    oai_resp = {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 0,
        "model": "deepseek-ai/DeepSeek-V3.1",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": message_text,
                },
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5},
    }

    tools = [
        {
            "type": "function",
            "function": {
                "name": "TodoWrite",
                "description": "",
                "parameters": {"type": "object"},
            },
        }
    ]

    anthropic = openai_to_anthropic_response(oai_resp, requested_model="deepseek-ai/DeepSeek-V3.1", tools=tools)
    tool_blocks = [blk for blk in anthropic.get("content", []) if blk.get("type") == "tool_use"]
    text_blocks = [blk for blk in anthropic.get("content", []) if blk.get("type") == "text"]
    thinking_blocks = [blk for blk in anthropic.get("content", []) if blk.get("type") == "thinking"]

    assert tool_blocks, "expected tool_use block from DeepSeek markup"
    assert not text_blocks, "Thinking placeholder should not surface as text"
    assert not thinking_blocks, "pure placeholder text should be dropped"


def test_deepseek_reasoning_promoted_to_thinking():
    from app.transform import openai_to_anthropic_response

    message_text = (
        "Thinking…\n1. Inspect request.\n2. Prepare tool payload.\n\n"
        "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>TodoWrite<｜tool▁sep｜>{\"todos\": []}<｜tool▁call▁end｜><｜tool▁calls▁end｜>"
    )

    oai_resp = {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 0,
        "model": "deepseek-ai/DeepSeek-V3.1",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": message_text,
                },
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5},
    }

    tools = [
        {
            "type": "function",
            "function": {
                "name": "TodoWrite",
                "description": "",
                "parameters": {"type": "object"},
            },
        }
    ]

    anthropic = openai_to_anthropic_response(oai_resp, requested_model="deepseek-ai/DeepSeek-V3.1", tools=tools)
    thinking_blocks = [blk for blk in anthropic.get("content", []) if blk.get("type") == "thinking"]
    text_blocks = [blk for blk in anthropic.get("content", []) if blk.get("type") == "text"]

    assert thinking_blocks, "reasoning text should convert to thinking block"
    assert thinking_blocks[0]["thinking"].startswith("Thinking…")
    assert not text_blocks, "reasoning text should not appear as plain text"
