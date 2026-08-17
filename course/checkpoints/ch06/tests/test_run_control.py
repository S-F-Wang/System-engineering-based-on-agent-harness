from __future__ import annotations

import asyncio
import sys

import pytest

from agent_harness import (
    AgentMessage,
    AgentRuntime,
    AgentSession,
    AsyncioProcessOperations,
    EventType,
    InputKind,
    ModelEnd,
    ModelRequest,
    ModelSpec,
    Role,
    RunGuard,
    SessionBusyError,
    StopReason,
    TerminalStatus,
    TextDelta,
    Tool,
    ToolCallDelta,
    ToolErrorCode,
    ToolResult,
    ToolResultMessage,
    Usage,
    UsageUpdate,
)


def test_session_rejects_a_second_prompt_while_one_run_is_active() -> None:
    class BlockingAdapter:
        def __init__(self) -> None:
            self.started = asyncio.Event()
            self.release = asyncio.Event()

        async def stream(self, request: ModelRequest):
            self.started.set()
            await self.release.wait()
            yield TextDelta("first complete")
            yield ModelEnd(StopReason.COMPLETE)

    async def scenario() -> None:
        adapter = BlockingAdapter()
        session = AgentSession(
            AgentRuntime(adapter, ModelSpec("scripted/one-active-run"))
        )
        first = session.start("first")
        await adapter.started.wait()

        assert session.busy is True
        with pytest.raises(SessionBusyError, match="active Run"):
            await session.prompt("second")

        adapter.release.set()
        result = await first.result()

        assert result.outcome.message == AgentMessage.text(
            Role.ASSISTANT, "first complete"
        )
        assert session.busy is False

    asyncio.run(scenario())


def test_steering_is_consumed_at_the_next_turn_boundary() -> None:
    class SteerableAdapter:
        def __init__(self) -> None:
            self.requests: list[ModelRequest] = []
            self.first_started = asyncio.Event()
            self.release_first = asyncio.Event()

        async def stream(self, request: ModelRequest):
            self.requests.append(request)
            if len(self.requests) == 1:
                self.first_started.set()
                await self.release_first.wait()
                yield TextDelta("initial direction")
            else:
                yield TextDelta("redirected result")
            yield ModelEnd(StopReason.COMPLETE)

    async def scenario() -> None:
        adapter = SteerableAdapter()
        session = AgentSession(
            AgentRuntime(adapter, ModelSpec("scripted/steering"))
        )
        handle = session.start("begin")
        await adapter.first_started.wait()

        session.steer("use the safer route")
        adapter.release_first.set()
        result = await handle.result()

        assert result.outcome.message == AgentMessage.text(
            Role.ASSISTANT, "redirected result"
        )
        assert len(adapter.requests) == 2
        assert adapter.requests[1].messages[-1] == AgentMessage.text(
            Role.USER, "use the safer route"
        ).to_model()
        assert result.pending_inputs == ()

    asyncio.run(scenario())


def test_follow_up_starts_a_new_run_only_after_the_current_run_settles() -> None:
    class FollowUpAdapter:
        def __init__(self) -> None:
            self.requests: list[ModelRequest] = []
            self.first_started = asyncio.Event()
            self.release_first = asyncio.Event()

        async def stream(self, request: ModelRequest):
            self.requests.append(request)
            if len(self.requests) == 1:
                self.first_started.set()
                await self.release_first.wait()
                yield TextDelta("first settled")
            else:
                yield TextDelta("follow-up settled")
            yield ModelEnd(StopReason.COMPLETE)

    async def scenario() -> None:
        adapter = FollowUpAdapter()
        session = AgentSession(
            AgentRuntime(adapter, ModelSpec("scripted/follow-up"))
        )
        handle = session.start("first")
        await adapter.first_started.wait()

        session.follow_up("then inspect the result")
        await asyncio.sleep(0)
        assert len(adapter.requests) == 1

        adapter.release_first.set()
        result = await handle.result()

        assert [outcome.message for outcome in result.outcomes] == [
            AgentMessage.text(Role.ASSISTANT, "first settled"),
            AgentMessage.text(Role.ASSISTANT, "follow-up settled"),
        ]
        assert adapter.requests[1].messages[-2:] == (
            AgentMessage.text(Role.ASSISTANT, "first settled").to_model(),
            AgentMessage.text(Role.USER, "then inspect the result").to_model(),
        )
        assert result.pending_inputs == ()

    asyncio.run(scenario())


def test_different_sessions_run_independently() -> None:
    class GateAdapter:
        def __init__(self, answer: str) -> None:
            self.answer = answer
            self.started = asyncio.Event()
            self.release = asyncio.Event()

        async def stream(self, request: ModelRequest):
            self.started.set()
            await self.release.wait()
            yield TextDelta(self.answer)
            yield ModelEnd(StopReason.COMPLETE)

    async def scenario() -> None:
        left_adapter = GateAdapter("left")
        right_adapter = GateAdapter("right")
        left = AgentSession(
            AgentRuntime(left_adapter, ModelSpec("scripted/left"))
        )
        right = AgentSession(
            AgentRuntime(right_adapter, ModelSpec("scripted/right"))
        )

        left_handle = left.start("one")
        right_handle = right.start("two")
        await asyncio.gather(left_adapter.started.wait(), right_adapter.started.wait())

        right_adapter.release.set()
        right_result = await asyncio.wait_for(right_handle.result(), timeout=1)
        assert right_result.outcome.message == AgentMessage.text(Role.ASSISTANT, "right")
        assert left.busy is True

        left_adapter.release.set()
        left_result = await asyncio.wait_for(left_handle.result(), timeout=1)
        assert left_result.outcome.message == AgentMessage.text(Role.ASSISTANT, "left")

    asyncio.run(scenario())


def test_cancellation_settles_parallel_tools_and_returns_unconsumed_input() -> None:
    async def scenario() -> None:
        quick_finished = asyncio.Event()
        slow_started = asyncio.Event()

        async def quick(arguments: dict[str, object]) -> ToolResult:
            quick_finished.set()
            return ToolResult("quick result")

        async def slow(arguments: dict[str, object]) -> ToolResult:
            slow_started.set()
            await asyncio.Event().wait()
            return ToolResult("unreachable")

        adapter = type(
            "ToolAdapter",
            (),
            {
                "stream": lambda self, request: _tool_call_stream(),
            },
        )()

        runtime = AgentRuntime(
            adapter,
            ModelSpec("scripted/cancel-tools"),
            tools=[
                Tool("quick", "Finish quickly", {"type": "object"}, quick),
                Tool("slow", "Wait for cancellation", {"type": "object"}, slow),
            ],
        )
        session = AgentSession(runtime)
        handle = session.start("run both")
        await asyncio.gather(quick_finished.wait(), slow_started.wait())
        await asyncio.sleep(0)

        session.steer("queued steering")
        session.follow_up("queued follow-up")
        events_task = asyncio.create_task(
            _collect_cancellation_events(handle)
        )
        handle.cancel()
        result = await asyncio.wait_for(handle.result(), timeout=1)
        events = await events_task

        assert result.outcome.stop_reason is StopReason.ABORTED
        assert result.outcome.status is TerminalStatus.CANCELLED
        tool_messages = [
            item for item in runtime.history if isinstance(item, ToolResultMessage)
        ]
        assert [item.tool_name for item in tool_messages] == ["quick", "slow"]
        assert tool_messages[0].result == ToolResult("quick result")
        assert tool_messages[1].result.error_code is ToolErrorCode.CANCELLED
        slow_end = next(
            event
            for event in events
            if event.type is EventType.TOOL_CALL_END
            and event.tool_call_id == "slow-id"
        )
        assert slow_end.tool_result is not None
        assert slow_end.tool_result.error_code is ToolErrorCode.CANCELLED
        assert runtime.history[-1] == result.outcome.message
        assert [item.kind for item in result.pending_inputs] == [
            InputKind.STEERING,
            InputKind.FOLLOW_UP,
        ]
        assert [item.message for item in result.pending_inputs] == [
            AgentMessage.text(Role.USER, "queued steering"),
            AgentMessage.text(Role.USER, "queued follow-up"),
        ]

    async def _tool_call_stream():
        yield ToolCallDelta(0, "quick-id", "quick", "{}")
        yield ToolCallDelta(1, "slow-id", "slow", "{}")
        yield ModelEnd(StopReason.TOOL_USE)

    async def _collect_cancellation_events(handle):
        return [event async for event in handle.events()]

    asyncio.run(scenario())


def test_cancellation_terminates_a_real_process_operation(tmp_path) -> None:
    async def scenario() -> None:
        started = tmp_path / "started"
        incorrectly_finished = tmp_path / "finished"
        script = (
            "from pathlib import Path; import time; "
            "Path('started').write_text('yes'); "
            "time.sleep(0.3); Path('finished').write_text('should not happen')"
        )
        operation = AsyncioProcessOperations()
        task = asyncio.create_task(
            operation.run([sys.executable, "-c", script], cwd=tmp_path)
        )
        for _ in range(100):
            if started.exists():
                break
            await asyncio.sleep(0.01)
        assert started.exists()

        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        await asyncio.sleep(0.35)

        assert not incorrectly_finished.exists()

    asyncio.run(scenario())


@pytest.mark.parametrize(
    ("guard", "expected"),
    [
        (RunGuard(max_turns=1), TerminalStatus.MAX_TURNS),
        (RunGuard(max_tool_calls=1), TerminalStatus.MAX_TOOL_CALLS),
        (RunGuard(max_total_tokens=5), TerminalStatus.MAX_TOTAL_TOKENS),
    ],
)
def test_run_guards_stop_with_distinct_status_at_a_settled_boundary(
    guard: RunGuard,
    expected: TerminalStatus,
) -> None:
    class RepeatingToolAdapter:
        def __init__(self) -> None:
            self.turn = 0

        async def stream(self, request: ModelRequest):
            self.turn += 1
            yield ToolCallDelta(self.turn - 1, f"call-{self.turn}", "step", "{}")
            yield UsageUpdate(Usage(4, 1, 5))
            yield ModelEnd(StopReason.TOOL_USE)

    async def scenario() -> None:
        async def step(arguments: dict[str, object]) -> ToolResult:
            return ToolResult("settled")

        adapter = RepeatingToolAdapter()
        runtime = AgentRuntime(
            adapter,
            ModelSpec("scripted/guard"),
            tools=[Tool("step", "Take one step", {"type": "object"}, step)],
            run_guard=guard,
        )

        outcome = await runtime.run([AgentMessage.text(Role.USER, "continue")])

        assert outcome.status is expected
        assert outcome.stop_reason is StopReason.ABORTED
        assert adapter.turn == 1
        assert isinstance(runtime.history[-1], ToolResultMessage)
        assert runtime.history[-1].tool_call_id == "call-1"

    asyncio.run(scenario())


def test_timeout_guard_cancels_provider_work_and_settles_distinctly() -> None:
    class BlockingAdapter:
        async def stream(self, request: ModelRequest):
            yield TextDelta("partial")
            await asyncio.Event().wait()
            yield ModelEnd(StopReason.COMPLETE)

    async def scenario() -> None:
        runtime = AgentRuntime(
            BlockingAdapter(),
            ModelSpec("scripted/timeout"),
            run_guard=RunGuard(timeout_seconds=0.02),
        )

        outcome = await asyncio.wait_for(
            runtime.run([AgentMessage.text(Role.USER, "wait")]), timeout=0.5
        )

        assert outcome.status is TerminalStatus.TIMEOUT
        assert outcome.stop_reason is StopReason.ABORTED
        assert outcome.message == AgentMessage.text(Role.ASSISTANT, "partial")
        assert runtime.history[-1] == outcome.message

    asyncio.run(scenario())


def test_session_handle_observes_follow_up_runs_in_lifecycle_order() -> None:
    class TwoRunAdapter:
        def __init__(self) -> None:
            self.turn = 0
            self.started = asyncio.Event()
            self.release = asyncio.Event()

        async def stream(self, request: ModelRequest):
            self.turn += 1
            if self.turn == 1:
                self.started.set()
                await self.release.wait()
            yield TextDelta(f"answer {self.turn}")
            yield ModelEnd(StopReason.COMPLETE)

    async def scenario() -> None:
        adapter = TwoRunAdapter()
        session = AgentSession(
            AgentRuntime(adapter, ModelSpec("scripted/session-events"))
        )
        handle = session.start("first")
        await adapter.started.wait()
        session.follow_up("second")
        events_task = asyncio.create_task(_collect(handle))
        adapter.release.set()

        await handle.result()
        event_types = [event.type for event in await events_task]

        first_end = event_types.index(EventType.AGENT_END)
        second_start = event_types.index(EventType.AGENT_START, first_end + 1)
        assert first_end < second_start
        assert event_types.count(EventType.AGENT_START) == 2
        assert event_types.count(EventType.AGENT_END) == 2

    async def _collect(handle):
        return [event async for event in handle.events()]

    asyncio.run(scenario())


def test_session_immediate_cancel_still_settles_the_accepted_run() -> None:
    class BlockingAdapter:
        async def stream(self, request: ModelRequest):
            await asyncio.Event().wait()
            yield ModelEnd(StopReason.COMPLETE)

    async def scenario() -> None:
        session = AgentSession(
            AgentRuntime(BlockingAdapter(), ModelSpec("scripted/session-cancel"))
        )
        handle = session.start("cancel now")

        handle.cancel()
        result = await asyncio.wait_for(handle.result(), timeout=0.2)

        assert result.outcome.status is TerminalStatus.CANCELLED
        assert result.outcome.stop_reason is StopReason.ABORTED
        assert session.busy is False

    asyncio.run(scenario())


def test_completed_handle_cannot_cancel_a_later_run() -> None:
    class TwoRunAdapter:
        def __init__(self) -> None:
            self.turn = 0
            self.second_started = asyncio.Event()
            self.release_second = asyncio.Event()

        async def stream(self, request: ModelRequest):
            self.turn += 1
            if self.turn == 2:
                self.second_started.set()
                await self.release_second.wait()
            yield TextDelta(f"answer {self.turn}")
            yield ModelEnd(StopReason.COMPLETE)

    async def scenario() -> None:
        adapter = TwoRunAdapter()
        session = AgentSession(
            AgentRuntime(adapter, ModelSpec("scripted/stale-handle"))
        )
        completed_handle = session.start("first")
        await completed_handle.result()

        later_handle = session.start("second")
        await adapter.second_started.wait()
        completed_handle.cancel()
        await asyncio.sleep(0)

        assert session.busy is True
        adapter.release_second.set()
        later = await later_handle.result()
        assert later.outcome.status is TerminalStatus.COMPLETED
        assert later.outcome.message == AgentMessage.text(Role.ASSISTANT, "answer 2")

    asyncio.run(scenario())
