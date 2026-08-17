from __future__ import annotations

import asyncio
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from threading import Thread
from typing import Iterator

import pytest

from agent_harness import (
    AgentMessage,
    ModelAdapterError,
    ModelErrorCode,
    ModelSpec,
    OpenAICompatibleAdapter,
    OpenAICompatibleConfig,
    Role,
    StopReason,
    TextContent,
    ToolCallContent,
    Usage,
    complete,
)


def _chunk(delta, *, finish_reason=None):
    return {
        'id': 'chatcmpl-local',
        'object': 'chat.completion.chunk',
        'created': 1,
        'model': 'local-test',
        'choices': [
            {'index': 0, 'delta': delta, 'finish_reason': finish_reason}
        ],
    }


@contextmanager
def fake_openai_server() -> Iterator[tuple[str, list[dict[str, object]]]]:
    requests: list[dict[str, object]] = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            length = int(self.headers.get('content-length', '0'))
            body = json.loads(self.rfile.read(length))
            requests.append(
                {'path': self.path, 'authorization': self.headers.get('authorization'), 'body': body}
            )
            if body['model'] in {'reject-me', 'retry-me', 'overflow-me'}:
                error_code = (
                    'context_length_exceeded'
                    if body['model'] == 'overflow-me'
                    else None
                )
                payload = json.dumps(
                    {
                        'error': {
                            'message': 'SECRET provider detail',
                            'type': 'invalid_request_error',
                            'code': error_code,
                        }
                    }
                ).encode()
                status = (
                    429
                    if body['model'] == 'retry-me'
                    else 400
                    if body['model'] == 'overflow-me'
                    else 401
                )
                self.send_response(status)
                self.send_header('content-type', 'application/json')
                self.send_header('content-length', str(len(payload)))
                if status == 429:
                    self.send_header('retry-after', '7')
                self.end_headers()
                self.wfile.write(payload)
                return

            events = [
                _chunk({'role': 'assistant', 'content': 'Hel'}),
                _chunk({'content': 'lo'}),
                _chunk(
                    {
                        'tool_calls': [
                            {
                                'index': 0,
                                'id': 'call-1',
                                'type': 'function',
                                'function': {'name': 'read', 'arguments': '{"path":'},
                            }
                        ]
                    }
                ),
                _chunk(
                    {'tool_calls': [{'index': 0, 'function': {'arguments': '"README.md"}'}}]}
                ),
                _chunk({}, finish_reason='tool_calls'),
                {
                    'id': 'chatcmpl-local',
                    'object': 'chat.completion.chunk',
                    'created': 1,
                    'model': 'local-test',
                    'choices': [],
                    'usage': {'prompt_tokens': 4, 'completion_tokens': 6, 'total_tokens': 10},
                },
            ]
            payload = ''.join(f'data: {json.dumps(event)}\n\n' for event in events)
            payload += 'data: [DONE]\n\n'
            encoded = payload.encode()
            self.send_response(200)
            self.send_header('content-type', 'text/event-stream')
            self.send_header('content-length', str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def log_message(self, format, *args):
            return

    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address
        yield f'http://{host}:{port}/v1', requests
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_openai_compatible_stream_is_translated_at_the_adapter_seam(monkeypatch):
    monkeypatch.setenv('OPENAI_API_KEY', 'ambient-key-must-not-win')
    with fake_openai_server() as (base_url, requests):
        adapter = OpenAICompatibleAdapter(
            OpenAICompatibleConfig(base_url=base_url, api_key='explicit-key')
        )
        result = asyncio.run(
            complete(
                adapter,
                [AgentMessage.text(Role.USER, 'inspect')],
                ModelSpec('local-test', max_output_tokens=128),
            )
        )

    assert result.message.content == (
        TextContent('Hello'),
        ToolCallContent('call-1', 'read', '{"path":"README.md"}'),
    )
    assert result.stop_reason is StopReason.TOOL_USE
    assert result.usage == Usage(4, 6, 10)
    assert requests[0]['path'] == '/v1/chat/completions'
    assert requests[0]['authorization'] == 'Bearer explicit-key'
    assert requests[0]['body']['messages'] == [{'role': 'user', 'content': 'inspect'}]


def test_openai_compatible_failure_is_normalized_without_raw_detail():
    with fake_openai_server() as (base_url, _):
        adapter = OpenAICompatibleAdapter(
            OpenAICompatibleConfig(base_url=base_url, api_key='explicit-key')
        )
        with pytest.raises(ModelAdapterError) as captured:
            asyncio.run(
                complete(
                    adapter,
                    [AgentMessage.text(Role.USER, 'fail safely')],
                    ModelSpec('reject-me'),
                )
            )

    assert captured.value.error.code is ModelErrorCode.AUTHENTICATION
    assert captured.value.error.status_code == 401
    assert captured.value.error.retryable is False
    assert 'SECRET' not in str(captured.value)


def test_adapter_surfaces_bounded_retry_hint_without_retrying_itself():
    with fake_openai_server() as (base_url, requests):
        adapter = OpenAICompatibleAdapter(
            OpenAICompatibleConfig(base_url=base_url, api_key='explicit-key')
        )
        with pytest.raises(ModelAdapterError) as captured:
            asyncio.run(
                complete(
                    adapter,
                    [AgentMessage.text(Role.USER, 'retry in runtime')],
                    ModelSpec('retry-me'),
                )
            )

    assert captured.value.error.code is ModelErrorCode.RATE_LIMIT
    assert captured.value.error.retry_after_seconds == 7.0
    assert len(requests) == 1


def test_adapter_normalizes_context_capacity_errors_for_compaction_recovery():
    with fake_openai_server() as (base_url, requests):
        adapter = OpenAICompatibleAdapter(
            OpenAICompatibleConfig(base_url=base_url, api_key='explicit-key')
        )
        with pytest.raises(ModelAdapterError) as captured:
            asyncio.run(
                complete(
                    adapter,
                    [AgentMessage.text(Role.USER, 'overflow safely')],
                    ModelSpec('overflow-me'),
                )
            )

    assert captured.value.error.code is ModelErrorCode.CONTEXT_OVERFLOW
    assert captured.value.error.retryable is False
    assert captured.value.error.status_code == 400
    assert len(requests) == 1
