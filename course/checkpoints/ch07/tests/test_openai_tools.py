from __future__ import annotations

import asyncio
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from threading import Thread
from typing import Iterator

from agent_harness import (
    AgentMessage,
    AgentRuntime,
    ModelSpec,
    OpenAICompatibleAdapter,
    OpenAICompatibleConfig,
    Role,
    StopReason,
    Tool,
    ToolResult,
)


def _chunk(delta: dict[str, object], *, finish_reason: str | None = None) -> dict[str, object]:
    return {
        "id": "chatcmpl-tools",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": "local-tools",
        "choices": [
            {"index": 0, "delta": delta, "finish_reason": finish_reason}
        ],
    }


@contextmanager
def tool_server() -> Iterator[tuple[str, list[dict[str, object]]]]:
    requests: list[dict[str, object]] = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            length = int(self.headers.get("content-length", "0"))
            requests.append(json.loads(self.rfile.read(length)))
            if len(requests) == 1:
                events = [
                    _chunk(
                        {
                            "role": "assistant",
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call-weather",
                                    "type": "function",
                                    "function": {
                                        "name": "weather",
                                        "arguments": '{"city":"Paris"}',
                                    },
                                }
                            ],
                        }
                    ),
                    _chunk({}, finish_reason="tool_calls"),
                ]
            else:
                events = [
                    _chunk({"role": "assistant", "content": "Paris is clear."}),
                    _chunk({}, finish_reason="stop"),
                ]
            payload = "".join(
                f"data: {json.dumps(event)}\n\n" for event in events
            ) + "data: [DONE]\n\n"
            encoded = payload.encode()
            self.send_response(200)
            self.send_header("content-type", "text/event-stream")
            self.send_header("content-length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def log_message(self, format: str, *args: object) -> None:
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address
        yield f"http://{host}:{port}/v1", requests
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_openai_compatible_adapter_translates_tool_definitions_and_results() -> None:
    async def weather(arguments: dict[str, object]) -> ToolResult:
        return ToolResult("clear")

    with tool_server() as (base_url, requests):
        runtime = AgentRuntime(
            OpenAICompatibleAdapter(
                OpenAICompatibleConfig(base_url=base_url, api_key="explicit-key")
            ),
            ModelSpec("local-tools"),
            tools=[
                Tool(
                    "weather",
                    "Get weather",
                    {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                    },
                    weather,
                )
            ],
        )
        outcome = asyncio.run(
            runtime.run([AgentMessage.text(Role.USER, "weather in Paris")])
        )

    assert outcome.stop_reason is StopReason.COMPLETE
    assert requests[0]["tools"] == [
        {
            "type": "function",
            "function": {
                "name": "weather",
                "description": "Get weather",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            },
        }
    ]
    assert requests[1]["messages"][-1] == {
        "role": "tool",
        "tool_call_id": "call-weather",
        "content": "clear",
    }
