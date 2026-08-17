from __future__ import annotations

import asyncio
from dataclasses import dataclass

import pytest

from agent_harness import (
    AgentMessage,
    ModelEnd,
    ModelMessage,
    ModelSpec,
    Role,
    ScriptedModelAdapter,
    StopReason,
    TextContent,
    TextDelta,
    ToolCallContent,
    ToolCallDelta,
    UnsupportedContentError,
    Usage,
    UsageUpdate,
    complete,
    to_model_messages,
)


def test_scripted_completion_crosses_the_typed_model_seam():
    adapter = ScriptedModelAdapter(
        [TextDelta('typed answer'), UsageUpdate(Usage(2, 3, 5)), ModelEnd(StopReason.COMPLETE)]
    )
    source = AgentMessage.text(Role.USER, 'hello')

    result = asyncio.run(complete(adapter, [source], ModelSpec('scripted/test')))

    assert result.message == ModelMessage(Role.ASSISTANT, (TextContent('typed answer'),))
    assert result.usage == Usage(2, 3, 5)
    request = adapter.received_requests[0]
    assert request.messages == (ModelMessage(Role.USER, (TextContent('hello'),)),)
    assert not isinstance(request.messages[0], dict)


def test_scripted_tool_argument_deltas_are_assembled_in_call_order():
    adapter = ScriptedModelAdapter(
        [
            ToolCallDelta(1, 'call-b', 'read', '{"path":'),
            ToolCallDelta(0, 'call-a', 'list', '{}'),
            ToolCallDelta(1, arguments_delta='"README.md"}'),
            ModelEnd(StopReason.TOOL_USE),
        ]
    )

    result = asyncio.run(complete(adapter, [], ModelSpec('scripted/tools')))

    assert result.message.content == (
        ToolCallContent('call-a', 'list', '{}'),
        ToolCallContent('call-b', 'read', '{"path":"README.md"}'),
    )


@dataclass(frozen=True)
class ImageContent:
    source: str = 'provider-image'
    schema_version: int = 1


def test_unsupported_content_fails_before_adapter_invocation():
    adapter = ScriptedModelAdapter([ModelEnd(StopReason.COMPLETE)])
    message = AgentMessage(Role.USER, (ImageContent(),))  # type: ignore[arg-type]

    with pytest.raises(UnsupportedContentError, match='only TextContent and ToolCallContent'):
        asyncio.run(complete(adapter, [message], ModelSpec('scripted/test')))

    assert adapter.received_requests == ()


def test_tool_calls_are_rejected_for_non_assistant_roles():
    message = AgentMessage(Role.USER, (ToolCallContent('call-1', 'read', '{}'),))

    with pytest.raises(UnsupportedContentError, match='assistant'):
        to_model_messages([message])
