from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass

import pytest

from agent_harness import (
    AgentMessage,
    AgentRuntime,
    EventType,
    ModelEnd,
    ModelAdapterError,
    ModelError,
    ModelErrorCode,
    ModelEvent,
    ModelRequest,
    ModelMessage,
    ModelSpec,
    Role,
    RetryPolicy,
    ScriptedModelAdapter,
    StopReason,
    TextContent,
    TextDelta,
    Usage,
    UsageUpdate,
    UnsupportedContentError,
)


def test_started_run_streams_ordered_events_and_settles_typed_outcome() -> None:
    async def scenario() -> None:
        prompt = AgentMessage.text(Role.USER, "hello runtime")
        adapter = ScriptedModelAdapter(
            [
                TextDelta("streamed "),
                TextDelta("answer"),
                UsageUpdate(Usage(3, 2, 5)),
                ModelEnd(StopReason.COMPLETE),
            ]
        )
        runtime = AgentRuntime(adapter, ModelSpec("scripted/runtime"))

        handle = runtime.start([prompt])
        events = [event async for event in handle.events()]
        outcome = await handle.result()

        assert [event.sequence for event in events] == list(range(1, len(events) + 1))
        assert [event.type for event in events] == [
            EventType.AGENT_START,
            EventType.MODEL_ATTEMPT_START,
            EventType.MODEL_EVENT,
            EventType.MODEL_EVENT,
            EventType.MODEL_EVENT,
            EventType.MODEL_EVENT,
            EventType.MESSAGE_END,
            EventType.AGENT_END,
        ]
        assert outcome.message == AgentMessage(
            Role.ASSISTANT, (TextContent("streamed answer"),)
        )
        assert outcome.stop_reason is StopReason.COMPLETE
        assert outcome.usage == Usage(3, 2, 5)
        assert outcome.error is None
        assert runtime.history == (prompt, outcome.message)
        assert adapter.received_requests[0].messages == (
            ModelMessage(Role.USER, (TextContent("hello runtime"),)),
        )

    asyncio.run(scenario())


def test_provider_failure_after_acceptance_settles_partial_assistant_outcome() -> None:
    class FailingAdapter:
        async def stream(self, request: ModelRequest) -> AsyncIterator[ModelEvent]:
            yield TextDelta("partial answer")
            yield UsageUpdate(Usage(7, 2, 9))
            raise ModelAdapterError(
                ModelError(
                    ModelErrorCode.SERVER,
                    "OpenAI-compatible request failed with status 503",
                    retryable=True,
                    status_code=503,
                )
            )

    async def scenario() -> None:
        prompt = AgentMessage.text(Role.USER, "fail coherently")
        runtime = AgentRuntime(
            FailingAdapter(),
            ModelSpec("scripted/failure"),
            retry_policy=RetryPolicy(delays=()),
        )

        handle = runtime.start([prompt])
        events = [event async for event in handle.events()]
        outcome = await handle.result()

        assert outcome.stop_reason is StopReason.ERROR
        assert outcome.message == AgentMessage.text(Role.ASSISTANT, "partial answer")
        assert outcome.usage == Usage(7, 2, 9)
        assert outcome.error == ModelError(
            ModelErrorCode.SERVER,
            "OpenAI-compatible request failed with status 503",
            retryable=True,
            status_code=503,
        )
        failure = next(
            event for event in events if event.type is EventType.MODEL_ATTEMPT_FAILED
        )
        assert failure.partial_text == "partial answer"
        assert failure.partial_usage == Usage(7, 2, 9)
        assert events[-2].type is EventType.MESSAGE_END
        assert events[-1].type is EventType.AGENT_END
        assert runtime.history == (prompt, outcome.message)

    asyncio.run(scenario())


def test_transient_failures_retry_on_the_default_two_four_eight_schedule() -> None:
    class AttemptAdapter:
        def __init__(self) -> None:
            self.attempt = 0

        async def stream(self, request: ModelRequest) -> AsyncIterator[ModelEvent]:
            self.attempt += 1
            if self.attempt <= 3:
                yield TextDelta(f"discard attempt {self.attempt}")
                code, status = (
                    (ModelErrorCode.TIMEOUT, 408),
                    (ModelErrorCode.RATE_LIMIT, 429),
                    (ModelErrorCode.SERVER, 503),
                )[self.attempt - 1]
                raise ModelAdapterError(
                    ModelError(code, f"transient {status}", True, status)
                )
            yield TextDelta("final answer")
            yield ModelEnd(StopReason.COMPLETE)

    async def scenario() -> None:
        delays: list[float] = []

        async def record_sleep(delay: float) -> None:
            delays.append(delay)

        adapter = AttemptAdapter()
        runtime = AgentRuntime(
            adapter,
            ModelSpec("scripted/retries"),
            sleeper=record_sleep,
        )

        handle = runtime.start([AgentMessage.text(Role.USER, "retry safely")])
        events = [event async for event in handle.events()]
        outcome = await handle.result()

        assert delays == [2.0, 4.0, 8.0]
        assert adapter.attempt == 4
        assert outcome.message == AgentMessage.text(Role.ASSISTANT, "final answer")
        assert outcome.attempts == 4
        assert [
            event.partial_text
            for event in events
            if event.type is EventType.MODEL_ATTEMPT_FAILED
        ] == ["discard attempt 1", "discard attempt 2", "discard attempt 3"]
        assert [
            message
            for message in runtime.history
            if message.role is Role.ASSISTANT
        ] == [outcome.message]

    asyncio.run(scenario())


def test_retry_after_is_honored_only_within_the_bounded_policy() -> None:
    class RetryAfterAdapter:
        def __init__(self, retry_after: float) -> None:
            self.retry_after = retry_after
            self.attempt = 0

        async def stream(self, request: ModelRequest) -> AsyncIterator[ModelEvent]:
            self.attempt += 1
            if self.attempt == 1:
                raise ModelAdapterError(
                    ModelError(
                        ModelErrorCode.RATE_LIMIT,
                        "rate limited",
                        True,
                        429,
                        self.retry_after,
                    )
                )
            yield TextDelta("recovered")
            yield ModelEnd(StopReason.COMPLETE)

    async def run_case(retry_after: float) -> tuple[list[float], int, AssistantOutcome]:
        delays: list[float] = []

        async def record_sleep(delay: float) -> None:
            delays.append(delay)

        adapter = RetryAfterAdapter(retry_after)
        outcome = await AgentRuntime(
            adapter,
            ModelSpec("scripted/retry-after"),
            sleeper=record_sleep,
        ).run([AgentMessage.text(Role.USER, "respect server hint")])
        return delays, adapter.attempt, outcome

    async def scenario() -> None:
        bounded = await run_case(12.5)
        excessive = await run_case(60.1)

        assert bounded[:2] == ([12.5], 2)
        assert bounded[2].stop_reason is StopReason.COMPLETE
        assert excessive[:2] == ([], 1)
        assert excessive[2].stop_reason is StopReason.ERROR
        assert excessive[2].error is not None
        assert excessive[2].error.retry_after_seconds == 60.1

    asyncio.run(scenario())


def test_runtime_classifies_retryability_from_normalized_error_codes() -> None:
    class ClassifiedAdapter:
        def __init__(self, error: ModelError) -> None:
            self.error = error
            self.attempt = 0

        async def stream(self, request: ModelRequest) -> AsyncIterator[ModelEvent]:
            self.attempt += 1
            if self.attempt == 1:
                raise ModelAdapterError(self.error)
            yield TextDelta("recovered")
            yield ModelEnd(StopReason.COMPLETE)

    async def run_case(error: ModelError) -> tuple[int, list[float], AssistantOutcome]:
        delays: list[float] = []

        async def record_sleep(delay: float) -> None:
            delays.append(delay)

        adapter = ClassifiedAdapter(error)
        outcome = await AgentRuntime(
            adapter,
            ModelSpec("scripted/classification"),
            sleeper=record_sleep,
        ).run([AgentMessage.text(Role.USER, "classify")])
        return adapter.attempt, delays, outcome

    async def scenario() -> None:
        connection = await run_case(
            ModelError(ModelErrorCode.CONNECTION, "connection failed", False)
        )
        authentication = await run_case(
            ModelError(ModelErrorCode.AUTHENTICATION, "bad credential", True, 401)
        )
        request = await run_case(
            ModelError(ModelErrorCode.REQUEST, "invalid request", True, 400)
        )

        assert connection[:2] == (2, [2.0])
        assert connection[2].stop_reason is StopReason.COMPLETE
        assert authentication[:2] == (1, [])
        assert authentication[2].stop_reason is StopReason.ERROR
        assert request[:2] == (1, [])
        assert request[2].stop_reason is StopReason.ERROR

    asyncio.run(scenario())


def test_stream_schema_failure_settles_without_retrying_or_leaking_exception() -> None:
    async def scenario() -> None:
        delays: list[float] = []

        async def record_sleep(delay: float) -> None:
            delays.append(delay)

        runtime = AgentRuntime(
            ScriptedModelAdapter([TextDelta("incomplete")]),
            ModelSpec("scripted/schema-failure"),
            sleeper=record_sleep,
        )

        outcome = await runtime.run(
            [AgentMessage.text(Role.USER, "settle malformed stream")]
        )

        assert outcome.stop_reason is StopReason.ERROR
        assert outcome.message == AgentMessage.text(Role.ASSISTANT, "incomplete")
        assert outcome.error is not None
        assert outcome.error.code is ModelErrorCode.SCHEMA
        assert outcome.error.retryable is False
        assert outcome.attempts == 1
        assert delays == []

    asyncio.run(scenario())


def test_cancellation_settles_partial_provider_work_as_aborted_outcome() -> None:
    class BlockingAdapter:
        def __init__(self) -> None:
            self.started = asyncio.Event()

        async def stream(self, request: ModelRequest) -> AsyncIterator[ModelEvent]:
            yield TextDelta("before cancel")
            yield UsageUpdate(Usage(4, 2, 6))
            self.started.set()
            await asyncio.Event().wait()
            yield ModelEnd(StopReason.COMPLETE)

    async def scenario() -> None:
        adapter = BlockingAdapter()
        prompt = AgentMessage.text(Role.USER, "cancel this")
        runtime = AgentRuntime(adapter, ModelSpec("scripted/cancel"))
        handle = runtime.start([prompt])
        await adapter.started.wait()

        handle.cancel()
        events = [event async for event in handle.events()]
        outcome = await handle.result()

        assert outcome.stop_reason is StopReason.ABORTED
        assert outcome.message == AgentMessage.text(Role.ASSISTANT, "before cancel")
        assert outcome.usage == Usage(4, 2, 6)
        assert outcome.error is None
        assert EventType.RUN_CANCELLED in [event.type for event in events]
        assert events[-2].type is EventType.MESSAGE_END
        assert events[-1].type is EventType.AGENT_END
        assert runtime.history == (prompt, outcome.message)

    asyncio.run(scenario())


def test_cancellation_interrupts_retry_wait_without_committing_failed_partial() -> None:
    class TransientAdapter:
        def __init__(self) -> None:
            self.attempt = 0

        async def stream(self, request: ModelRequest) -> AsyncIterator[ModelEvent]:
            self.attempt += 1
            yield TextDelta("diagnostic only")
            raise ModelAdapterError(
                ModelError(ModelErrorCode.CONNECTION, "connection failed", True)
            )

    async def scenario() -> None:
        sleeping = asyncio.Event()

        async def blocking_sleep(delay: float) -> None:
            sleeping.set()
            await asyncio.Event().wait()

        adapter = TransientAdapter()
        runtime = AgentRuntime(
            adapter,
            ModelSpec("scripted/cancel-backoff"),
            sleeper=blocking_sleep,
        )
        handle = runtime.start([AgentMessage.text(Role.USER, "cancel retry")])
        await sleeping.wait()

        handle.cancel()
        events = [event async for event in handle.events()]
        outcome = await handle.result()

        assert adapter.attempt == 1
        assert outcome.stop_reason is StopReason.ABORTED
        assert outcome.message == AgentMessage.text(Role.ASSISTANT, "")
        failure = next(
            event for event in events if event.type is EventType.MODEL_ATTEMPT_FAILED
        )
        assert failure.partial_text == "diagnostic only"
        assert runtime.history[-1] == outcome.message

    asyncio.run(scenario())


def test_invalid_input_fails_before_run_acceptance_without_mutating_history() -> None:
    @dataclass(frozen=True)
    class ImageContent:
        source: str = "unsupported"
        schema_version: int = 1

    async def scenario() -> None:
        adapter = ScriptedModelAdapter([ModelEnd(StopReason.COMPLETE)])
        runtime = AgentRuntime(adapter, ModelSpec("scripted/preflight"))
        invalid = AgentMessage(Role.USER, (ImageContent(),))  # type: ignore[arg-type]

        with pytest.raises(UnsupportedContentError):
            runtime.start([invalid])

        assert runtime.history == ()
        assert adapter.received_requests == ()

    asyncio.run(scenario())


def test_aggregate_run_guard_is_opt_in() -> None:
    adapter = ScriptedModelAdapter([ModelEnd(StopReason.COMPLETE)])

    unbounded = AgentRuntime(adapter, ModelSpec("scripted/unbounded"))
    explicit_guard = object()
    bounded = AgentRuntime(
        adapter,
        ModelSpec("scripted/bounded"),
        run_guard=explicit_guard,
    )

    assert unbounded.run_guard is None
    assert bounded.run_guard is explicit_guard


def test_immediate_cancellation_still_reaches_a_settled_boundary() -> None:
    class NeverStartedAdapter:
        async def stream(self, request: ModelRequest) -> AsyncIterator[ModelEvent]:
            await asyncio.Event().wait()
            yield ModelEnd(StopReason.COMPLETE)

    async def scenario() -> None:
        runtime = AgentRuntime(
            NeverStartedAdapter(), ModelSpec("scripted/immediate-cancel")
        )
        handle = runtime.start([AgentMessage.text(Role.USER, "cancel immediately")])

        handle.cancel()
        outcome = await asyncio.wait_for(handle.result(), timeout=1)
        events = [event async for event in handle.events()]

        assert outcome.stop_reason is StopReason.ABORTED
        assert events[-1].type is EventType.AGENT_END
        assert runtime.history[-1] == outcome.message

    asyncio.run(scenario())


def test_data_after_model_end_is_a_non_retryable_schema_outcome() -> None:
    async def scenario() -> None:
        runtime = AgentRuntime(
            ScriptedModelAdapter(
                [ModelEnd(StopReason.COMPLETE), TextDelta("invalid late data")]
            ),
            ModelSpec("scripted/late-data"),
        )

        outcome = await runtime.run(
            [AgentMessage.text(Role.USER, "validate event order")]
        )

        assert outcome.stop_reason is StopReason.ERROR
        assert outcome.error is not None
        assert outcome.error.code is ModelErrorCode.SCHEMA
        assert outcome.attempts == 1

    asyncio.run(scenario())


def test_invalid_runtime_dependencies_fail_during_construction() -> None:
    adapter = ScriptedModelAdapter([ModelEnd(StopReason.COMPLETE)])
    model = ModelSpec("scripted/invalid-construction")

    with pytest.raises(TypeError, match="retry_policy"):
        AgentRuntime(adapter, model, retry_policy="invalid")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="sleeper"):
        AgentRuntime(adapter, model, sleeper=None)  # type: ignore[arg-type]

    assert adapter.received_requests == ()
