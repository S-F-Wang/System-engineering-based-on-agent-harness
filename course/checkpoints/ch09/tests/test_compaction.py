from __future__ import annotations

import asyncio

import pytest

from agent_harness import (
    AgentMessage,
    AgentRuntime,
    AgentSession,
    CompactionPolicy,
    CompactionStrategy,
    CompactionTrigger,
    CompactionWarningCode,
    EventType,
    JSONLSessionStore,
    MemorySessionStore,
    ModelEnd,
    ModelAdapterError,
    ModelError,
    ModelErrorCode,
    ModelOperation,
    ModelSpec,
    Role,
    ScriptedModelAdapter,
    StopReason,
    StructuredSummary,
    TextDelta,
    ToolCallContent,
    ToolCallDelta,
    Tool,
    ToolResult,
    ToolResultMessage,
    Usage,
    UsageUpdate,
    create_agent_session,
    migrate_session_file,
)


class WordTokenEstimator:
    def estimate(self, messages) -> int:
        return sum(
            len(block.text.split())
            for message in messages
            for block in getattr(message, "content", ())
            if hasattr(block, "text")
        )


def test_manual_compaction_persists_a_checkpoint_and_changes_only_model_context() -> None:
    async def scenario() -> None:
        original = (
            AgentMessage.text(Role.USER, "old question with details"),
            AgentMessage.text(Role.ASSISTANT, "old answer with rationale"),
            AgentMessage.text(Role.USER, "recent question"),
            AgentMessage.text(Role.ASSISTANT, "recent answer"),
        )
        store = MemorySessionStore()
        state = store.create("manual-compaction")
        with store.writer(state.session_id) as writer:
            writer.append(original)
        adapter = ScriptedModelAdapter(
            [
                [
                    TextDelta("Decided to retain the safe migration path."),
                    UsageUpdate(Usage(9, 8, 17)),
                    ModelEnd(StopReason.COMPLETE),
                ],
                [TextDelta("continued from checkpoint"), ModelEnd(StopReason.COMPLETE)],
            ]
        )
        session = create_agent_session(
            AgentRuntime(adapter, ModelSpec("scripted/compact")),
            store=store,
            session_id=state.session_id,
            compaction_policy=CompactionPolicy(keep_recent_tokens=2),
            token_estimator=WordTokenEstimator(),
        )

        compacted = await session.compact("preserve migration decisions")

        assert compacted.checkpoint.trigger is CompactionTrigger.MANUAL
        assert compacted.checkpoint.summary == StructuredSummary(
            "Decided to retain the safe migration path.",
            focus="preserve migration decisions",
        )
        assert compacted.checkpoint.tokens_before == 12
        assert compacted.checkpoint.summary_usage == Usage(9, 8, 17)
        assert compacted.checkpoint.retained_tail == (
            AgentMessage.text(Role.ASSISTANT, "recent answer"),
        )
        recovered = store.read(state.session_id)
        assert recovered.history() == original
        assert recovered.compactions() == (compacted.checkpoint,)

        await session.run("what next?")

        assert adapter.received_requests[0].operation is ModelOperation.COMPACTION
        continuation = adapter.received_requests[1]
        assert continuation.operation is ModelOperation.RUN
        assert continuation.messages == (
            *compacted.checkpoint.model_context(),
            AgentMessage.text(Role.USER, "what next?").to_model(),
        )
        assert store.read(state.session_id).history() == (
            *original,
            AgentMessage.text(Role.USER, "what next?"),
            AgentMessage.text(Role.ASSISTANT, "continued from checkpoint"),
        )

    asyncio.run(scenario())


@pytest.mark.parametrize(
    ("model", "reserve", "keep", "threshold"),
    [
        (ModelSpec("small", context_window=32_768), 4_096, 8_192, 28_672),
        (ModelSpec("broad", context_window=131_072), 16_384, 20_000, 114_688),
        (ModelSpec("unknown"), None, 20_000, None),
    ],
)
def test_compaction_policy_resolves_adaptive_small_and_broad_context_defaults(
    model: ModelSpec,
    reserve: int | None,
    keep: int,
    threshold: int | None,
) -> None:
    resolved = CompactionPolicy().resolve(model)

    assert resolved.reserve_tokens == reserve
    assert resolved.keep_recent_tokens == keep
    assert resolved.threshold_tokens == threshold


def test_retained_tail_keeps_an_oversized_tool_batch_structurally_complete() -> None:
    class UnitEstimator:
        def estimate(self, messages) -> int:
            return len(messages)

    tool_call = AgentMessage(
        Role.ASSISTANT,
        (ToolCallContent("call-1", "lookup", '{}'),),
    )
    tool_result = ToolResultMessage("call-1", "lookup", ToolResult("large result"))

    plan = CompactionStrategy().plan(
        (
            AgentMessage.text(Role.USER, "summarize this prefix"),
            tool_call,
            tool_result,
        ),
        keep_recent_tokens=1,
        estimator=UnitEstimator(),
    )

    assert plan.source == (AgentMessage.text(Role.USER, "summarize this prefix"),)
    assert plan.retained_tail == (tool_call, tool_result)


def test_threshold_compaction_runs_after_the_turn_settles() -> None:
    async def scenario() -> None:
        adapter = ScriptedModelAdapter(
            [
                [TextDelta("settled answer"), ModelEnd(StopReason.COMPLETE)],
                [TextDelta("threshold summary"), ModelEnd(StopReason.COMPLETE)],
            ]
        )
        store = MemorySessionStore()
        session = create_agent_session(
            AgentRuntime(
                adapter,
                ModelSpec(
                    "scripted/threshold",
                    context_window=20,
                    max_output_tokens=4,
                ),
            ),
            store=store,
            compaction_policy=CompactionPolicy(),
            token_estimator=WordTokenEstimator(),
        )

        result = await session.run(" ".join(f"word-{index}" for index in range(20)))

        assert result.outcome.stop_reason is StopReason.COMPLETE
        assert store.read(session.session_id).compactions()[0].trigger is (
            CompactionTrigger.THRESHOLD
        )
        assert [request.operation for request in adapter.received_requests] == [
            ModelOperation.RUN,
            ModelOperation.COMPACTION,
        ]

    asyncio.run(scenario())


def test_failed_threshold_compaction_warns_without_changing_settled_history() -> None:
    class ThresholdFailureAdapter:
        def __init__(self) -> None:
            self.requests = []

        async def stream(self, request):
            self.requests.append(request)
            if request.operation is ModelOperation.COMPACTION:
                raise ModelAdapterError(
                    ModelError(
                        ModelErrorCode.PROVIDER,
                        "summary unavailable",
                        retryable=False,
                    )
                )
            yield TextDelta("settled answer")
            yield ModelEnd(StopReason.COMPLETE)

    async def scenario() -> None:
        adapter = ThresholdFailureAdapter()
        store = MemorySessionStore()
        session = create_agent_session(
            AgentRuntime(
                adapter,
                ModelSpec(
                    "scripted/threshold-failure",
                    context_window=20,
                    max_output_tokens=4,
                ),
            ),
            store=store,
            token_estimator=WordTokenEstimator(),
        )
        prompt = " ".join(f"word-{index}" for index in range(20))

        result = await session.run(prompt)

        assert result.outcome.stop_reason is StopReason.COMPLETE
        state = store.read(session.session_id)
        assert state.history() == (
            AgentMessage.text(Role.USER, prompt),
            AgentMessage.text(Role.ASSISTANT, "settled answer"),
        )
        assert state.compactions() == ()
        assert session.warnings[-1].code is CompactionWarningCode.THRESHOLD_FAILED
        assert "history unchanged" in session.warnings[-1].message

    asyncio.run(scenario())


def test_context_overflow_compacts_once_and_retries_outside_transient_backoff() -> None:
    class OverflowThenSuccessAdapter:
        def __init__(self) -> None:
            self.requests = []
            self.run_attempts = 0

        async def stream(self, request):
            self.requests.append(request)
            if request.operation is ModelOperation.COMPACTION:
                yield TextDelta("overflow recovery summary")
                yield ModelEnd(StopReason.COMPLETE)
                return
            self.run_attempts += 1
            if self.run_attempts == 1:
                raise ModelAdapterError(
                    ModelError(
                        ModelErrorCode.CONTEXT_OVERFLOW,
                        "context capacity exceeded",
                        retryable=False,
                    )
                )
            yield TextDelta("recovered answer")
            yield ModelEnd(StopReason.COMPLETE)

    async def scenario() -> None:
        adapter = OverflowThenSuccessAdapter()
        sleeps: list[float] = []

        async def sleeper(delay: float) -> None:
            sleeps.append(delay)

        store = MemorySessionStore()
        state = store.create("overflow-recovery")
        with store.writer(state.session_id) as writer:
            writer.append(
                (
                    AgentMessage.text(Role.USER, "old question"),
                    AgentMessage.text(Role.ASSISTANT, "old answer"),
                )
            )
        session = create_agent_session(
            AgentRuntime(
                adapter,
                ModelSpec("scripted/overflow"),
                sleeper=sleeper,
            ),
            store=store,
            session_id=state.session_id,
            compaction_policy=CompactionPolicy(keep_recent_tokens=1),
            token_estimator=WordTokenEstimator(),
        )

        handle = session.start("continue")
        events_task = asyncio.create_task(_collect_events(handle))
        result = await handle.result()
        events = await events_task

        assert result.outcome.message == AgentMessage.text(
            Role.ASSISTANT, "recovered answer"
        )
        assert result.outcome.attempts == 2
        assert sleeps == []
        assert [request.operation for request in adapter.requests] == [
            ModelOperation.RUN,
            ModelOperation.COMPACTION,
            ModelOperation.RUN,
        ]
        assert [checkpoint.trigger for checkpoint in store.read(state.session_id).compactions()] == [
            CompactionTrigger.OVERFLOW
        ]
        assert EventType.COMPACTION_START in [event.type for event in events]
        assert EventType.COMPACTION_END in [event.type for event in events]

    async def _collect_events(handle):
        return [event async for event in handle.events()]

    asyncio.run(scenario())


def test_failed_overflow_compaction_terminates_the_run_clearly() -> None:
    class FailedRecoveryAdapter:
        def __init__(self) -> None:
            self.requests = []

        async def stream(self, request):
            self.requests.append(request)
            if request.operation is ModelOperation.COMPACTION:
                raise ModelAdapterError(
                    ModelError(
                        ModelErrorCode.PROVIDER,
                        "summary unavailable",
                        retryable=False,
                    )
                )
            raise ModelAdapterError(
                ModelError(
                    ModelErrorCode.CONTEXT_OVERFLOW,
                    "context capacity exceeded",
                    retryable=False,
                )
            )
            yield  # pragma: no cover - keeps this an async generator

    async def scenario() -> None:
        adapter = FailedRecoveryAdapter()
        store = MemorySessionStore()
        state = store.create("failed-overflow-recovery")
        with store.writer(state.session_id) as writer:
            writer.append(
                (
                    AgentMessage.text(Role.USER, "old question"),
                    AgentMessage.text(Role.ASSISTANT, "old answer"),
                )
            )
        session = create_agent_session(
            AgentRuntime(adapter, ModelSpec("scripted/failed-overflow")),
            store=store,
            session_id=state.session_id,
            compaction_policy=CompactionPolicy(keep_recent_tokens=1),
            token_estimator=WordTokenEstimator(),
        )

        result = await session.run("continue")

        assert result.outcome.stop_reason is StopReason.ERROR
        assert result.outcome.error is not None
        assert result.outcome.error.code is ModelErrorCode.COMPACTION_FAILED
        assert "Compaction failed" in result.outcome.error.message
        assert [request.operation for request in adapter.requests] == [
            ModelOperation.RUN,
            ModelOperation.COMPACTION,
        ]
        assert store.read(state.session_id).compactions() == ()

    asyncio.run(scenario())


def test_manual_compaction_uses_session_cancellation_and_commits_nothing() -> None:
    class BlockingSummaryAdapter:
        def __init__(self) -> None:
            self.started = asyncio.Event()

        async def stream(self, request):
            assert request.operation is ModelOperation.COMPACTION
            self.started.set()
            await asyncio.Event().wait()
            yield ModelEnd(StopReason.COMPLETE)

    async def scenario() -> None:
        adapter = BlockingSummaryAdapter()
        store = MemorySessionStore()
        state = store.create("cancel-compaction")
        original = (
            AgentMessage.text(Role.USER, "old question"),
            AgentMessage.text(Role.ASSISTANT, "old answer"),
        )
        with store.writer(state.session_id) as writer:
            writer.append(original)
        session = create_agent_session(
            AgentRuntime(adapter, ModelSpec("scripted/cancel-compaction")),
            store=store,
            session_id=state.session_id,
            compaction_policy=CompactionPolicy(keep_recent_tokens=1),
            token_estimator=WordTokenEstimator(),
        )
        task = asyncio.create_task(session.compact())
        await adapter.started.wait()

        session.cancel()

        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=0.2)
        assert session.busy is False
        assert store.read(state.session_id).history() == original
        assert store.read(state.session_id).compactions() == ()

    asyncio.run(scenario())


def test_jsonl_compaction_checkpoint_reopens_with_materialized_effective_context(
    tmp_path,
) -> None:
    async def scenario() -> None:
        store = JSONLSessionStore(tmp_path)
        state = store.create("durable-compaction")
        original = (
            AgentMessage.text(Role.USER, "old question"),
            AgentMessage.text(Role.ASSISTANT, "recent answer"),
        )
        with store.writer(state.session_id) as writer:
            writer.append(original)
        session = create_agent_session(
            AgentRuntime(
                ScriptedModelAdapter(
                    [TextDelta("durable summary"), ModelEnd(StopReason.COMPLETE)]
                ),
                ModelSpec("scripted/durable-compaction"),
            ),
            store=store,
            session_id=state.session_id,
            compaction_policy=CompactionPolicy(keep_recent_tokens=2),
            token_estimator=WordTokenEstimator(),
        )

        result = await session.compact()
        reopened = JSONLSessionStore(tmp_path).read(state.session_id)

        assert reopened.history() == original
        assert reopened.compactions() == (result.checkpoint,)
        assert reopened.effective_history() == (
            result.checkpoint.summary.as_message(),
            *result.checkpoint.retained_tail,
        )

        continuation_adapter = ScriptedModelAdapter(
            [TextDelta("continued"), ModelEnd(StopReason.COMPLETE)]
        )
        continued = create_agent_session(
            AgentRuntime(
                continuation_adapter,
                ModelSpec("scripted/reopen-compaction"),
            ),
            store=JSONLSessionStore(tmp_path),
            session_id=state.session_id,
        )
        await continued.run("next")
        assert continuation_adapter.received_requests[0].messages == (
            *result.checkpoint.model_context(),
            AgentMessage.text(Role.USER, "next").to_model(),
        )

        source = store.path_for(state.session_id)
        original_bytes = source.read_bytes()
        destination = tmp_path / "migrated" / "sessions" / source.name
        migrate_session_file(source, destination)
        migrated = JSONLSessionStore(tmp_path / "migrated").read(state.session_id)
        assert source.read_bytes() == original_bytes
        assert migrated.history() == JSONLSessionStore(tmp_path).read(
            state.session_id
        ).history()
        assert migrated.compactions() == (result.checkpoint,)

    asyncio.run(scenario())


def test_unknown_context_window_disables_threshold_compaction_with_one_warning() -> None:
    async def scenario() -> None:
        adapter = ScriptedModelAdapter(
            [
                [TextDelta("first"), ModelEnd(StopReason.COMPLETE)],
                [TextDelta("second"), ModelEnd(StopReason.COMPLETE)],
            ]
        )
        session = AgentSession(
            AgentRuntime(adapter, ModelSpec("scripted/unknown-window")),
            token_estimator=WordTokenEstimator(),
        )

        await session.run("a long first prompt")
        await session.run("a long second prompt")

        assert [warning.code for warning in session.warnings] == [
            CompactionWarningCode.CONTEXT_WINDOW_UNKNOWN
        ]
        assert [request.operation for request in adapter.received_requests] == [
            ModelOperation.RUN,
            ModelOperation.RUN,
        ]

    asyncio.run(scenario())


def test_summary_generation_reuses_runtime_retry_policy_and_sleeper() -> None:
    class RetrySummaryAdapter:
        def __init__(self) -> None:
            self.requests = []

        async def stream(self, request):
            self.requests.append(request)
            if len(self.requests) == 1:
                raise ModelAdapterError(
                    ModelError(
                        ModelErrorCode.RATE_LIMIT,
                        "retry summary",
                        retryable=True,
                    )
                )
            yield TextDelta("summary after retry")
            yield ModelEnd(StopReason.COMPLETE)

    async def scenario() -> None:
        adapter = RetrySummaryAdapter()
        sleeps: list[float] = []

        async def sleeper(delay: float) -> None:
            sleeps.append(delay)

        runtime = AgentRuntime(
            adapter,
            ModelSpec("scripted/retry-summary"),
            sleeper=sleeper,
        )
        runtime.restore_history(
            (
                AgentMessage.text(Role.USER, "old question"),
                AgentMessage.text(Role.ASSISTANT, "old answer"),
            )
        )
        session = AgentSession(
            runtime,
            compaction_policy=CompactionPolicy(keep_recent_tokens=1),
            token_estimator=WordTokenEstimator(),
        )

        result = await session.compact()

        assert result.checkpoint.summary.text == "summary after retry"
        assert sleeps == [2.0]
        assert [request.operation for request in adapter.requests] == [
            ModelOperation.COMPACTION,
            ModelOperation.COMPACTION,
        ]

    asyncio.run(scenario())


def test_a_second_context_overflow_terminates_without_another_compaction() -> None:
    class RepeatedOverflowAdapter:
        def __init__(self) -> None:
            self.requests = []

        async def stream(self, request):
            self.requests.append(request)
            if request.operation is ModelOperation.COMPACTION:
                yield TextDelta("one recovery summary")
                yield ModelEnd(StopReason.COMPLETE)
                return
            raise ModelAdapterError(
                ModelError(
                    ModelErrorCode.CONTEXT_OVERFLOW,
                    "still too large",
                    retryable=False,
                )
            )
            yield  # pragma: no cover - keeps this an async generator

    async def scenario() -> None:
        adapter = RepeatedOverflowAdapter()
        runtime = AgentRuntime(adapter, ModelSpec("scripted/repeated-overflow"))
        runtime.restore_history(
            (
                AgentMessage.text(Role.USER, "old question"),
                AgentMessage.text(Role.ASSISTANT, "old answer"),
            )
        )
        session = AgentSession(
            runtime,
            compaction_policy=CompactionPolicy(keep_recent_tokens=1),
            token_estimator=WordTokenEstimator(),
        )

        result = await session.run("continue")

        assert result.outcome.stop_reason is StopReason.ERROR
        assert result.outcome.error is not None
        assert result.outcome.error.code is ModelErrorCode.CONTEXT_OVERFLOW
        assert [request.operation for request in adapter.requests] == [
            ModelOperation.RUN,
            ModelOperation.COMPACTION,
            ModelOperation.RUN,
        ]

    asyncio.run(scenario())


def test_threshold_check_compacts_a_settled_tool_turn_before_model_continuation() -> None:
    class WeightedEstimator:
        def estimate(self, messages) -> int:
            total = 0
            for message in messages:
                if isinstance(message, ToolResultMessage):
                    total += 1
                    continue
                text = " ".join(
                    block.text
                    for block in message.content
                    if hasattr(block, "text")
                )
                total += 10 if "large-prefix" in text else 1
            return total

    class ToolThenAnswerAdapter:
        def __init__(self) -> None:
            self.requests = []
            self.run_turn = 0

        async def stream(self, request):
            self.requests.append(request)
            if request.operation is ModelOperation.COMPACTION:
                yield TextDelta("short summary")
                yield ModelEnd(StopReason.COMPLETE)
                return
            self.run_turn += 1
            if self.run_turn == 1:
                yield ToolCallDelta(0, "call-1", "lookup", '{}')
                yield ModelEnd(StopReason.TOOL_USE)
                return
            yield TextDelta("final answer")
            yield ModelEnd(StopReason.COMPLETE)

    async def scenario() -> None:
        async def lookup(arguments: dict[str, object]) -> ToolResult:
            return ToolResult("result")

        adapter = ToolThenAnswerAdapter()
        session = AgentSession(
            AgentRuntime(
                adapter,
                ModelSpec(
                    "scripted/threshold-tool-turn",
                    context_window=6,
                    max_output_tokens=1,
                ),
                tools=[Tool("lookup", "lookup", {"type": "object"}, lookup)],
            ),
            compaction_policy=CompactionPolicy(keep_recent_tokens=1),
            token_estimator=WeightedEstimator(),
        )

        result = await session.run("large-prefix")

        assert result.outcome.message == AgentMessage.text(
            Role.ASSISTANT, "final answer"
        )
        assert [request.operation for request in adapter.requests] == [
            ModelOperation.RUN,
            ModelOperation.COMPACTION,
            ModelOperation.RUN,
        ]
        assert "short summary" in adapter.requests[-1].messages[0].content[0].text

    asyncio.run(scenario())
