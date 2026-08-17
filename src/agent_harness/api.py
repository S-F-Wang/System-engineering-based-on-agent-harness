"""Concise public factories over explicit, replaceable harness infrastructure."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Mapping, Sequence
from pathlib import Path

from .coding import CodingToolPreset, create_coding_tool_preset
from .compaction import CompactionPolicy, CompactionStrategy, TokenEstimator
from .extensions import Extension
from .model import (
    AgentMessage,
    ModelAdapter,
    ModelSpec,
    OpenAICompatibleAdapter,
    OpenAICompatibleConfig,
    Role,
)
from .observability import (
    JSONLRunStore,
    LocalWorkspace,
    RunArtifactStore,
    StandardTraceRedactor,
    StandardTraceSink,
)
from .persistence import SessionStore
from .resources import PromptAssembler, ResourceLoader
from .runtime import AgentRuntime, RetryPolicy, RunGuard
from .session import AgentSession, RunObserver, create_agent_session
from .tools import Tool, ToolExecutor, ToolOutputBudget


Sleeper = Callable[[float], Awaitable[None]]


def _adapter(
    adapter_or_config: ModelAdapter | OpenAICompatibleConfig,
) -> tuple[ModelAdapter, tuple[str, ...]]:
    if isinstance(adapter_or_config, OpenAICompatibleConfig):
        return (
            OpenAICompatibleAdapter(adapter_or_config),
            (adapter_or_config.api_key, *adapter_or_config.headers.values()),
        )
    if not callable(getattr(adapter_or_config, "stream", None)):
        raise TypeError(
            "adapter_or_config must be a ModelAdapter or OpenAICompatibleConfig"
        )
    return adapter_or_config, ()


def _local_observation(
    workspace: Path,
    *,
    storage_root: str | Path | None,
    trace_secrets: Sequence[str],
) -> tuple[LocalWorkspace, StandardTraceSink, RunArtifactStore]:
    redactor = StandardTraceRedactor(secrets=trace_secrets)
    local = LocalWorkspace.open(
        workspace,
        storage_root=storage_root,
        redactor=redactor,
    )
    artifacts = RunArtifactStore(local.runs)
    return local, StandardTraceSink(
        local.runs,
        workspace=workspace,
        artifacts=artifacts,
    ), artifacts


def create_session(
    adapter_or_config: ModelAdapter | OpenAICompatibleConfig,
    model: ModelSpec,
    *,
    workspace: str | Path,
    storage_root: str | Path | None = None,
    session_store: SessionStore | None = None,
    run_store: JSONLRunStore | None = None,
    observer: RunObserver | None = None,
    session_id: str | None = None,
    fork_from: str | None = None,
    no_save: bool = False,
    strict_tracing: bool = False,
    trace_secrets: Sequence[str] = (),
    tools: Sequence[Tool] = (),
    tool_executor: ToolExecutor | None = None,
    tool_output_budget: ToolOutputBudget | None = None,
    retry_policy: RetryPolicy | None = None,
    sleeper: Sleeper = asyncio.sleep,
    run_guard: RunGuard | None = None,
    generation_settings: Mapping[str, object] | None = None,
    compaction_policy: CompactionPolicy | None = None,
    compaction_strategy: CompactionStrategy | None = None,
    token_estimator: TokenEstimator | None = None,
    extensions: Sequence[Extension] = (),
) -> AgentSession:
    """Create a general durable Session, or an explicit all-memory Session."""

    workspace_path = Path(workspace).resolve()
    if not workspace_path.is_dir():
        raise ValueError("workspace must be an existing directory")
    if not isinstance(model, ModelSpec):
        raise TypeError("model must be a ModelSpec")
    adapter, config_secrets = _adapter(adapter_or_config)
    if no_save and any(
        value is not None for value in (session_store, run_store, observer, session_id)
    ):
        raise ValueError("no_save disables Session, Trace, Annotation, and Artifact stores")
    effective_store = session_store
    effective_observer = observer
    if not no_save:
        if effective_store is None and run_store is None and effective_observer is None:
            local, effective_observer, _ = _local_observation(
                workspace_path,
                storage_root=storage_root,
                trace_secrets=(*config_secrets, *trace_secrets),
            )
            effective_store = local.sessions
        elif effective_observer is None and run_store is not None:
            effective_observer = StandardTraceSink(
                run_store,
                workspace=workspace_path,
            )
    runtime = AgentRuntime(
        adapter,
        model,
        tools=tools,
        tool_executor=tool_executor,
        tool_output_budget=tool_output_budget,
        retry_policy=retry_policy,
        sleeper=sleeper,
        run_guard=run_guard,
        generation_settings=generation_settings,
    )
    return create_agent_session(
        runtime,
        store=effective_store,
        session_id=session_id,
        fork_from=fork_from,
        no_save=no_save,
        compaction_policy=compaction_policy,
        compaction_strategy=compaction_strategy,
        token_estimator=token_estimator,
        extensions=extensions,
        observer=effective_observer,
        strict_tracing=strict_tracing,
    )


def create_coding_session(
    adapter_or_config: ModelAdapter | OpenAICompatibleConfig,
    model: ModelSpec,
    *,
    workspace: str | Path,
    storage_root: str | Path | None = None,
    session_store: SessionStore | None = None,
    run_store: JSONLRunStore | None = None,
    observer: RunObserver | None = None,
    session_id: str | None = None,
    fork_from: str | None = None,
    no_save: bool = False,
    strict_tracing: bool = False,
    trace_secrets: Sequence[str] = (),
    resource_loader: ResourceLoader | None = None,
    prompt_assembler: PromptAssembler | None = None,
    active_skills: Sequence[str] = (),
    preset: CodingToolPreset | None = None,
    retry_policy: RetryPolicy | None = None,
    sleeper: Sleeper = asyncio.sleep,
    run_guard: RunGuard | None = None,
    generation_settings: Mapping[str, object] | None = None,
    compaction_policy: CompactionPolicy | None = None,
    compaction_strategy: CompactionStrategy | None = None,
    token_estimator: TokenEstimator | None = None,
    extensions: Sequence[Extension] = (),
) -> AgentSession:
    """Create the explicit Coding Agent preset over the same AgentSession seam."""

    workspace_path = Path(workspace).resolve()
    if not workspace_path.is_dir():
        raise ValueError("workspace must be an existing directory")
    adapter, config_secrets = _adapter(adapter_or_config)
    if no_save and any(
        value is not None for value in (session_store, run_store, observer, session_id)
    ):
        raise ValueError("no_save disables Session, Trace, Annotation, and Artifact stores")
    effective_store = session_store
    effective_observer = observer
    artifact_store: RunArtifactStore | None = None
    if not no_save:
        if effective_store is None and run_store is None and effective_observer is None:
            local, effective_observer, artifact_store = _local_observation(
                workspace_path,
                storage_root=storage_root,
                trace_secrets=(*config_secrets, *trace_secrets),
            )
            effective_store = local.sessions
        elif effective_observer is None and run_store is not None:
            artifact_store = RunArtifactStore(run_store)
            effective_observer = StandardTraceSink(
                run_store,
                workspace=workspace_path,
                artifacts=artifact_store,
            )
    effective_preset = preset or create_coding_tool_preset(
        workspace_path,
        artifact_store=artifact_store,
    )
    if effective_preset.workspace.root != workspace_path:
        raise ValueError("Coding Tool Preset must use the Session workspace")
    loader = resource_loader or ResourceLoader(workspace_path)
    if loader.workspace != workspace_path:
        raise ValueError("ResourceLoader must use the Session workspace")
    resources = loader.load()
    assembler = prompt_assembler or PromptAssembler(
        "You are Omega, a coding agent. Work carefully inside the supplied "
        "workspace, use Tools for evidence, and verify changes before finishing."
    )
    prompt = assembler.assemble(
        tools=effective_preset.tools,
        resources=resources,
        active_skills=active_skills,
    )
    runtime = AgentRuntime(
        adapter,
        model,
        tools=effective_preset.tools,
        retry_policy=retry_policy,
        sleeper=sleeper,
        run_guard=run_guard,
        generation_settings=generation_settings,
        history=(AgentMessage.text(Role.SYSTEM, prompt.text),) if no_save else (),
    )
    session = create_agent_session(
        runtime,
        store=effective_store,
        session_id=session_id,
        fork_from=fork_from,
        no_save=no_save,
        compaction_policy=compaction_policy,
        compaction_strategy=compaction_strategy,
        token_estimator=token_estimator,
        extensions=extensions,
        prompt_hashes=prompt.prompt_hashes,
        resource_hashes=prompt.resource_hashes,
        observer=effective_observer,
        strict_tracing=strict_tracing,
    )
    if not no_save and session_id is None:
        # The effective system context is settled once into each new durable Session.
        assert effective_store is not None and session.session_id is not None
        with effective_store.writer(session.session_id) as writer:
            writer.append((AgentMessage.text(Role.SYSTEM, prompt.text),))
        runtime.restore_history((AgentMessage.text(Role.SYSTEM, prompt.text),))
    return session
