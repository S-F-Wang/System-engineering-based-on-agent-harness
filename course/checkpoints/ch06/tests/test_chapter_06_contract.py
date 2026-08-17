from __future__ import annotations

import asyncio

import pytest

from agent_harness import (
    AgentMessage,
    AgentRuntime,
    AgentSession,
    CompactionPolicy,
    ModelEnd,
    ModelRequest,
    ModelSpec,
    Role,
    SessionBusyError,
    StopReason,
    TextDelta,
)


def test_compaction_accepts_work_only_at_an_idle_settled_boundary() -> None:
    class BlockingAdapter:
        def __init__(self) -> None:
            self.started = asyncio.Event()
            self.release = asyncio.Event()

        async def stream(self, request: ModelRequest):
            self.started.set()
            await self.release.wait()
            yield TextDelta("settled")
            yield ModelEnd(StopReason.COMPLETE)

    async def scenario() -> None:
        adapter = BlockingAdapter()
        session = AgentSession(
            AgentRuntime(adapter, ModelSpec("scripted/settled-boundary"))
        )
        handle = session.start("active")
        await adapter.started.wait()

        with pytest.raises(SessionBusyError, match="Settled Boundary"):
            await session.compact()

        adapter.release.set()
        await handle.result()

    asyncio.run(scenario())


def test_compaction_configuration_fails_before_model_work_is_accepted() -> None:
    runtime = AgentRuntime(
        type("UnusedAdapter", (), {"stream": lambda self, request: None})(),
        ModelSpec("scripted/configuration"),
        history=(
            AgentMessage.text(Role.USER, "question"),
            AgentMessage.text(Role.ASSISTANT, "answer"),
        ),
    )
    session = AgentSession(runtime)

    with pytest.raises(ValueError, match="focus cannot be empty"):
        asyncio.run(session.compact("  "))
    with pytest.raises(ValueError, match="keep_recent_tokens"):
        CompactionPolicy(keep_recent_tokens=0)
