"""Provider-neutral messages, model events, and model adapters."""

from __future__ import annotations

from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
import json
from types import MappingProxyType
from typing import Literal, Protocol, TypeAlias, cast
from urllib.parse import urlparse

import openai


class Role(str, Enum):
    SYSTEM = 'system'
    USER = 'user'
    ASSISTANT = 'assistant'


class StopReason(str, Enum):
    COMPLETE = 'complete'
    TOOL_USE = 'tool_use'
    LENGTH = 'length'
    CONTENT_FILTER = 'content_filter'
    OTHER = 'other'


class ModelErrorCode(str, Enum):
    AUTHENTICATION = 'authentication'
    REQUEST = 'request'
    RATE_LIMIT = 'rate_limit'
    TIMEOUT = 'timeout'
    CONNECTION = 'connection'
    SERVER = 'server'
    PROVIDER = 'provider'


class UnsupportedContentError(ValueError):
    """Raised before provider I/O for a content variant outside version one."""


class ModelProtocolError(RuntimeError):
    """Raised when an adapter emits an incomplete provider-neutral stream."""


@dataclass(frozen=True, slots=True)
class TextContent:
    text: str
    type: Literal['text'] = field(default='text', init=False)
    schema_version: Literal[1] = field(default=1, init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.text, str):
            raise TypeError('TextContent.text must be a string')


@dataclass(frozen=True, slots=True)
class ToolCallContent:
    id: str
    name: str
    arguments: str
    type: Literal['tool_call'] = field(default='tool_call', init=False)
    schema_version: Literal[1] = field(default=1, init=False)

    def __post_init__(self) -> None:
        if not self.id or not self.name:
            raise ValueError('a Tool Call requires non-empty id and name')
        if not isinstance(self.arguments, str):
            raise TypeError('ToolCallContent.arguments must be a JSON string')


ContentBlock: TypeAlias = TextContent | ToolCallContent


@dataclass(frozen=True, slots=True)
class AgentMessage:
    role: Role
    content: tuple[ContentBlock, ...]

    @classmethod
    def text(cls, role: Role, text: str) -> 'AgentMessage':
        return cls(role=role, content=(TextContent(text),))


@dataclass(frozen=True, slots=True)
class ModelMessage:
    role: Role
    content: tuple[ContentBlock, ...]


@dataclass(frozen=True, slots=True)
class ModelSpec:
    model_id: str
    context_window: int | None = None
    max_output_tokens: int = 4096
    supports_tools: bool = True

    def __post_init__(self) -> None:
        if not self.model_id.strip():
            raise ValueError('ModelSpec.model_id cannot be empty')
        if self.context_window is not None and self.context_window <= 0:
            raise ValueError('context_window must be positive when supplied')
        if self.max_output_tokens <= 0:
            raise ValueError('max_output_tokens must be positive')


@dataclass(frozen=True, slots=True)
class Usage:
    input_tokens: int
    output_tokens: int
    total_tokens: int
    estimated: bool = False


@dataclass(frozen=True, slots=True)
class ModelRequest:
    messages: tuple[ModelMessage, ...]
    model: ModelSpec


@dataclass(frozen=True, slots=True)
class TextDelta:
    text: str


@dataclass(frozen=True, slots=True)
class ToolCallDelta:
    index: int
    id: str = ''
    name: str = ''
    arguments_delta: str = ''


@dataclass(frozen=True, slots=True)
class UsageUpdate:
    usage: Usage


@dataclass(frozen=True, slots=True)
class ModelEnd:
    stop_reason: StopReason


ModelEvent: TypeAlias = TextDelta | ToolCallDelta | UsageUpdate | ModelEnd


@dataclass(frozen=True, slots=True)
class ModelResult:
    message: ModelMessage
    stop_reason: StopReason
    usage: Usage | None = None


@dataclass(frozen=True, slots=True)
class ModelError:
    code: ModelErrorCode
    message: str
    retryable: bool
    status_code: int | None = None


class ModelAdapterError(RuntimeError):
    def __init__(self, error: ModelError) -> None:
        self.error = error
        super().__init__(error.message)


class ModelAdapter(Protocol):
    def stream(self, request: ModelRequest) -> AsyncIterator[ModelEvent]: ...


def _validate_content(block: object, role: Role) -> ContentBlock:
    if not isinstance(block, (TextContent, ToolCallContent)):
        raise UnsupportedContentError(
            'version one supports only TextContent and ToolCallContent'
        )
    if getattr(block, 'schema_version', None) != 1:
        raise UnsupportedContentError('unsupported Content Block schema version')
    if isinstance(block, ToolCallContent) and role is not Role.ASSISTANT:
        raise UnsupportedContentError('ToolCallContent is valid only for assistant messages')
    return block


def to_model_messages(messages: Sequence[AgentMessage]) -> tuple[ModelMessage, ...]:
    converted: list[ModelMessage] = []
    for message in messages:
        if not isinstance(message, AgentMessage):
            raise TypeError('model input must contain AgentMessage values')
        content = tuple(_validate_content(block, message.role) for block in message.content)
        converted.append(ModelMessage(role=message.role, content=content))
    return tuple(converted)


class ScriptedModelAdapter:
    """Replay a provider-neutral event script without credentials or I/O."""

    def __init__(self, events: Sequence[ModelEvent]) -> None:
        self._events = tuple(events)
        self._requests: list[ModelRequest] = []

    @property
    def received_requests(self) -> tuple[ModelRequest, ...]:
        return tuple(self._requests)

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelEvent]:
        self._requests.append(request)
        for event in self._events:
            yield event


async def complete(
    adapter: ModelAdapter,
    messages: Sequence[AgentMessage],
    model: ModelSpec,
) -> ModelResult:
    request = ModelRequest(to_model_messages(messages), model)
    text_parts: list[str] = []
    tool_drafts: dict[int, dict[str, str]] = {}
    usage: Usage | None = None
    end: ModelEnd | None = None
    async for event in adapter.stream(request):
        if end is not None:
            raise ModelProtocolError('an adapter emitted data after ModelEnd')
        if isinstance(event, TextDelta):
            text_parts.append(event.text)
        elif isinstance(event, ToolCallDelta):
            if event.index < 0:
                raise ModelProtocolError('Tool Call indexes cannot be negative')
            draft = tool_drafts.setdefault(
                event.index, {'id': '', 'name': '', 'arguments': ''}
            )
            draft['id'] += event.id
            draft['name'] += event.name
            draft['arguments'] += event.arguments_delta
        elif isinstance(event, UsageUpdate):
            usage = event.usage
        elif isinstance(event, ModelEnd):
            end = event
        else:
            raise ModelProtocolError(f'unsupported model event: {type(event).__name__}')
    if end is None:
        raise ModelProtocolError('an adapter stream must end with ModelEnd')
    blocks: list[ContentBlock] = []
    if text_parts:
        blocks.append(TextContent(''.join(text_parts)))
    for index in sorted(tool_drafts):
        draft = tool_drafts[index]
        try:
            blocks.append(ToolCallContent(**draft))
        except (TypeError, ValueError) as error:
            raise ModelProtocolError(f'incomplete Tool Call at index {index}') from error
    return ModelResult(
        message=ModelMessage(Role.ASSISTANT, tuple(blocks)),
        stop_reason=end.stop_reason,
        usage=usage,
    )


@dataclass(frozen=True, slots=True)
class OpenAICompatibleConfig:
    base_url: str
    api_key: str
    headers: Mapping[str, str] = field(default_factory=dict)
    extra_body: Mapping[str, object] = field(default_factory=dict)
    timeout_seconds: float = 60.0

    def __post_init__(self) -> None:
        parsed = urlparse(self.base_url)
        if parsed.scheme not in {'http', 'https'} or not parsed.netloc:
            raise ValueError('base_url must be an explicit HTTP(S) URL')
        if not self.api_key:
            raise ValueError('api_key must be supplied explicitly')
        if self.timeout_seconds <= 0:
            raise ValueError('timeout_seconds must be positive')
        object.__setattr__(self, 'headers', MappingProxyType(dict(self.headers)))
        object.__setattr__(self, 'extra_body', MappingProxyType(dict(self.extra_body)))


def _provider_message(message: ModelMessage) -> dict[str, object]:
    text = ''.join(block.text for block in message.content if isinstance(block, TextContent))
    tool_calls = [block for block in message.content if isinstance(block, ToolCallContent)]
    encoded: dict[str, object] = {'role': message.role.value, 'content': text or None}
    if tool_calls:
        encoded['tool_calls'] = [
            {
                'id': block.id,
                'type': 'function',
                'function': {'name': block.name, 'arguments': block.arguments},
            }
            for block in tool_calls
        ]
    return encoded


def _stop_reason(value: str | None) -> StopReason:
    if value is None:
        return StopReason.OTHER
    return {
        'stop': StopReason.COMPLETE,
        'tool_calls': StopReason.TOOL_USE,
        'length': StopReason.LENGTH,
        'content_filter': StopReason.CONTENT_FILTER,
    }.get(value, StopReason.OTHER)


def _normalized_error(error: Exception) -> ModelError:
    status = getattr(error, 'status_code', None)
    if isinstance(error, openai.AuthenticationError):
        code, retryable = ModelErrorCode.AUTHENTICATION, False
    elif isinstance(error, openai.RateLimitError) or status == 429:
        code, retryable = ModelErrorCode.RATE_LIMIT, True
    elif isinstance(error, openai.APITimeoutError) or status == 408:
        code, retryable = ModelErrorCode.TIMEOUT, True
    elif isinstance(error, openai.APIConnectionError):
        code, retryable = ModelErrorCode.CONNECTION, True
    elif isinstance(status, int) and status >= 500:
        code, retryable = ModelErrorCode.SERVER, True
    elif isinstance(error, (openai.BadRequestError, openai.NotFoundError)):
        code, retryable = ModelErrorCode.REQUEST, False
    else:
        code, retryable = ModelErrorCode.PROVIDER, False
    status_text = f' with status {status}' if status is not None else ''
    return ModelError(
        code=code,
        message=f'OpenAI-compatible request failed{status_text}',
        retryable=retryable,
        status_code=status,
    )


class OpenAICompatibleAdapter:
    """Translate the streaming Chat Completions protocol at one seam."""

    def __init__(self, config: OpenAICompatibleConfig) -> None:
        self._config = config
        self._client = openai.AsyncOpenAI(
            api_key=config.api_key,
            base_url=config.base_url.rstrip('/') + '/',
            default_headers=dict(config.headers),
            timeout=config.timeout_seconds,
            max_retries=0,
        )

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelEvent]:
        finish_reason: str | None = None
        try:
            response = await self._client.chat.completions.create(
                model=request.model.model_id,
                messages=cast(list, [_provider_message(item) for item in request.messages]),
                max_tokens=request.model.max_output_tokens,
                stream=True,
                stream_options={'include_usage': True},
                extra_body=dict(self._config.extra_body) or None,
            )
            async for chunk in response:
                if chunk.usage is not None:
                    input_tokens = chunk.usage.prompt_tokens or 0
                    output_tokens = chunk.usage.completion_tokens or 0
                    total_tokens = chunk.usage.total_tokens or input_tokens + output_tokens
                    yield UsageUpdate(Usage(input_tokens, output_tokens, total_tokens))
                for choice in chunk.choices:
                    delta = choice.delta
                    if delta.content:
                        yield TextDelta(delta.content)
                    for tool_call in delta.tool_calls or ():
                        function = tool_call.function
                        yield ToolCallDelta(
                            index=tool_call.index,
                            id=tool_call.id or '',
                            name=(function.name if function else None) or '',
                            arguments_delta=(function.arguments if function else None) or '',
                        )
                    if choice.finish_reason is not None:
                        finish_reason = choice.finish_reason
        except openai.OpenAIError as error:
            raise ModelAdapterError(_normalized_error(error)) from None
        yield ModelEnd(_stop_reason(finish_reason))
