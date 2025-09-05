from typing import List, Literal, Optional, Union, Dict, Any
from pydantic import BaseModel, Field


# Minimal Anthropic v1/messages schema (text-only)


class TextContent(BaseModel):
    type: Literal["text"] = "text"
    text: str


class MessageInput(BaseModel):
    role: Literal["user", "assistant", "system"]
    # Support either array-of-blocks (preferred) or a plain string (fallback)
    content: Union[str, List[Dict[str, Any]]]


class ToolDefinition(BaseModel):
    name: str
    description: Optional[str] = None
    input_schema: Dict[str, Any] = Field(default_factory=dict)


class MessagesRequest(BaseModel):
    model: str
    max_tokens: int = Field(..., ge=1)
    messages: List[MessageInput]
    system: Optional[Union[str, List[TextContent]]] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    stop_sequences: Optional[List[str]] = None
    stream: Optional[bool] = False
    metadata: Optional[Dict[str, Any]] = None
    tools: Optional[List[ToolDefinition]] = None


class Usage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0


class TextBlock(BaseModel):
    type: Literal["text"] = "text"
    text: str


class MessageResponse(BaseModel):
    id: str
    type: Literal["message"] = "message"
    role: Literal["assistant"] = "assistant"
    model: str
    content: List[TextBlock]
    stop_reason: Optional[Literal["end_turn", "max_tokens", "stop_sequence", "tool_use"]] = None
    stop_sequence: Optional[str] = None
    usage: Usage = Field(default_factory=Usage)


# Streaming event payloads (subset)


class MessageStartEvent(BaseModel):
    type: Literal["message_start"] = "message_start"
    message: MessageResponse


class ContentBlockStartEvent(BaseModel):
    type: Literal["content_block_start"] = "content_block_start"
    index: int
    content_block: TextBlock


class ContentBlockDeltaEvent(BaseModel):
    type: Literal["content_block_delta"] = "content_block_delta"
    index: int
    delta: Dict[str, str]  # {"type":"text_delta", "text":"..."}


class ContentBlockStopEvent(BaseModel):
    type: Literal["content_block_stop"] = "content_block_stop"
    index: int


class MessageDeltaEvent(BaseModel):
    type: Literal["message_delta"] = "message_delta"
    delta: Dict[str, Any]  # e.g., {"stop_reason": "end_turn", "usage": {"output_tokens": N}}
    usage: Optional[Usage] = None


class MessageStopEvent(BaseModel):
    type: Literal["message_stop"] = "message_stop"


class ErrorBody(BaseModel):
    type: str
    message: str


class ErrorResponse(BaseModel):
    type: Literal["error"] = "error"
    error: ErrorBody
