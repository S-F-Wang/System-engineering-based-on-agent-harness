from __future__ import annotations

import asyncio

import pytest

from agent_harness import AsyncioProcessOperations, RunGuard


def test_run_guard_and_process_inputs_fail_before_work_is_accepted() -> None:
    for name in ("max_turns", "max_tool_calls", "max_total_tokens"):
        with pytest.raises(ValueError, match=name):
            RunGuard(**{name: 0})
    with pytest.raises(ValueError, match="timeout_seconds"):
        RunGuard(timeout_seconds=0)

    async def scenario() -> None:
        operation = AsyncioProcessOperations()
        with pytest.raises(ValueError, match="command"):
            await operation.run([])
        with pytest.raises(ValueError, match="timeout_seconds"):
            await operation.run(["not-started"], timeout_seconds=0)

    asyncio.run(scenario())
