# Agent Harness Course

This context defines the shared language for a chapter-based course whose constructed result is a reusable agent harness.

## Language

**Chapter Notebook**:
The primary learning artifact for one coherent harness capability, combining its explanation with its construction.
_Avoid_: Lesson source, generated notebook

**Chapter Template**:
The required teaching sequence for every Chapter Notebook: motivation from the preceding Checkpoint, conceptual model, minimal execution, staged construction, observable trace, failure boundaries and trade-offs, Checkpoint export and verification, then public API summary.
_Avoid_: Exercise sheet, API reference dump

**Course Spine**:
The nine cumulative Chapter Increments that construct the Reusable Harness through model boundaries, runtime, tools, run control, durable sessions, context management, extensions, Coding Agent resources, and interfaces plus release.
_Avoid_: Setup-only chapter, one chapter per class

**Authoritative Course Tree**:
The single `course/` source hierarchy containing Chapter Notebooks, generated Checkpoint Packages, and course tooling; competing course prototypes remain historical inputs until post-gate archival.
_Avoid_: `courses/`, release copy, Raw Notebook

**Checkpoint**:
The immutable, runnable harness state reached at the end of a Chapter Notebook and used as the starting point for later chapters, without promising compatibility with later Checkpoints.
_Avoid_: Exercise solution, demo script

**Chapter Increment**:
The single coherent harness capability introduced by a Chapter Notebook relative to its preceding Checkpoint.
_Avoid_: Full rewrite, unrelated feature bundle

**Export Cell**:
A tagged code cell whose contents replace a declared Python module when the Chapter Notebook is exported into its Checkpoint.
_Avoid_: Narrative cell, append-only patch

**Checkpoint Package**:
The complete installable Python package snapshot produced for one Checkpoint by carrying forward unchanged modules and replacing modules declared by that chapter's Export Cells.
_Avoid_: Source fragment, manually maintained package copy

**Checkpoint Gate**:
The mandatory offline validation sequence that a generated Checkpoint Package must pass before publication, including notebook execution, compilation, installation, imports, and cumulative tests.
_Avoid_: Optional smoke test, manual inspection

**Reusable Harness**:
The library-first infrastructure artifact obtained from the completed course and consumed independently of its Chapter Notebooks or interactive shells.
_Avoid_: Notebook runtime, course demo

**Package Layer**:
One responsibility-focused subpackage inside the single `agent-harness` distribution, governed by explicit inward dependency rules rather than separate package publication.
_Avoid_: Independent distribution, arbitrary utility folder

**Pi-aligned**:
An architecture that follows Pi's separation of model, agent, session, resource, and interface responsibilities without promising API or protocol compatibility.
_Avoid_: Pi-compatible, Pi port

**AgentMessage**:
A message in the harness conversation that may carry model-facing content or harness-specific context.
_Avoid_: Provider message, API payload

**Content Block**:
A discriminated, provider-neutral unit inside a message; version one implements TextContent and ToolCallContent while preserving an extensible union for later modalities.
_Avoid_: Raw content string, provider-specific delta

**ModelMessage**:
A provider-neutral message that belongs to the language model context after harness-specific messages have been filtered or transformed.
_Avoid_: AgentMessage, OpenAI message

**ModelAdapter**:
The boundary that translates ModelMessages and model events between the Reusable Harness and one provider API.
_Avoid_: Agent, Provider client

**OpenAICompatibleAdapter**:
The sole production ModelAdapter in version one, translating harness contracts to streaming `/v1/chat/completions` requests through `openai.AsyncOpenAI` with an explicitly configurable endpoint.
_Avoid_: Responses API adapter, provider branch inside AgentRuntime

**OpenAICompatibleConfig**:
The explicit library configuration for the OpenAICompatibleAdapter, containing endpoint, credential, model, headers, and constrained request extensions without consulting ambient process configuration.
_Avoid_: Environment loader, global mutable settings

**ModelSpec**:
The explicit, provider-neutral description of one configured model's identifier, context capacity, output limit, and Tool capability, kept separate from transport credentials and endpoint settings.
_Avoid_: Provider catalog entry, inferred model-name behavior

**Token Estimate**:
A usage value marked as estimated when exact Provider accounting is unavailable, never merged indistinguishably with reported usage.
_Avoid_: Provider Usage, cost estimate

**ScriptedModelAdapter**:
The deterministic, offline ModelAdapter used by Chapter Notebooks and tests to replay specified model events without network access or credentials.
_Avoid_: Production provider, mock hidden inside AgentRuntime

**Stop Reason**:
The provider-neutral terminal classification of an assistant generation, such as completion, tool use, length, error, or abort.
_Avoid_: Provider finish reason, exception type

**Model Error**:
The normalized failure information retained when a model operation settles unsuccessfully after an Agent Run has begun.
_Avoid_: Configuration error, raw provider exception

**RetryPolicy**:
The AgentRuntime-owned classification and backoff policy for retrying transient model failures without duplicating retries inside ModelAdapter or applying automatic retries to Tools.
_Avoid_: Provider SDK retry, Compaction recovery

**Tool**:
A model-discoverable capability defined by an input contract and an execution entry point.
_Avoid_: Shell command, arbitrary function

**Tool Schema**:
The explicit JSON Schema contract for a Tool's model-supplied arguments, validated before execution and kept independent of any Python validation framework.
_Avoid_: Inferred function signature, provider wire object

**ToolResult**:
The structured outcome of one Tool execution, including model-facing content and optional harness-facing metadata.
_Avoid_: Output string, ToolResultMessage

**Tool Error**:
The normalized error ToolResultMessage produced by AgentRuntime after a Tool implementation, argument parser, schema validator, or executor fails, with diagnostic detail retained only in the Run Trace.
_Avoid_: Successful ToolResult, uncaught implementation exception

**Tool Output Budget**:
The model-context limit applied to one ToolResult, independently of the full sanitized output retained as a Run Artifact for a durable Run.
_Avoid_: Bash timeout, Run token guard

**Truncation Notice**:
The structured and model-visible description of a truncated ToolResult, including original extent, retained range, and truncation direction.
_Avoid_: Silent ellipsis, Run Artifact

**ToolExecutor**:
The boundary that runs a Tool in a particular local, isolated, or remote execution environment.
_Avoid_: Tool, permission policy

**Coding Tool Preset**:
The optional Coding Agent bundle of workspace-aware `read`, `write`, `edit`, and `bash` Tools, registered explicitly by library consumers and enabled by default by the Omega Interface Adapters.
_Avoid_: AgentRuntime defaults, mandatory tool set

**BashOperations**:
The replaceable Coding Agent process backend that executes commands through a real Bash binary with a fixed workspace, cancellation, timeout, and process-tree termination semantics.
_Avoid_: PowerShell compatibility shim, AgentRuntime subprocess code

**Tool Environment**:
The explicit environment mapping passed to a process-executing Tool after sensitive inherited variables have been removed and any required secret names have been deliberately allowed.
_Avoid_: Unfiltered process environment, sandbox

**Workspace Boundary**:
The resolved directory within which Coding Tool Preset file operations are permitted; it is a product-layer safety constraint rather than a core permission system.
_Avoid_: Process sandbox, permission prompt

**Atomic File Mutation**:
A write or edit operation staged in the destination directory and committed by atomic replacement after path and text validation, minimizing partially written workspace files.
_Avoid_: Bash write, transaction across multiple files

**Tool Batch**:
The ordered set of Tool calls requested by one assistant response and settled as part of the same model turn.
_Avoid_: Agent Run, unordered task group

**Extension**:
An explicitly registered module that adds or intercepts harness capabilities through the public extension contract.
_Avoid_: Built-in feature, automatically trusted project script

**Extension API**:
The bounded capability surface through which a trusted Extension registers Tools, passive Event subscribers, ordered Hooks, chat commands, lifecycle callbacks, and versioned Custom Entries without mutating Runtime internals.
_Avoid_: AgentRuntime internals, Provider registry

**Fail-safe Hook**:
An intervening Hook whose failure prevents the pending action or Run from proceeding with uncertain or unreviewed semantics, unlike a passive Event subscriber failure.
_Avoid_: Event listener, best-effort logger

**Custom Entry**:
A namespaced, versioned, JSON-serializable Session entry used to persist Extension state without automatically entering model context.
_Avoid_: AgentMessage, unversioned arbitrary object

**AgentResources**:
The explicit set of system context, Skills, and Prompt Templates made available to an AgentSession.
_Avoid_: ResourceLoader, project directory

**Skill**:
A named set of agent-facing instructions that can be deliberately activated for a task.
_Avoid_: Extension, Prompt Template

**Prompt Template**:
A named user-prompt pattern expanded with explicit arguments before an Agent Run begins.
_Avoid_: Skill, system prompt

**Prompt Assembly**:
The deterministic construction of effective system context from the built-in prompt, Tool guidance, Skill catalog, hierarchical Context Files, and appended system instructions, unless a trusted replacement system prompt is selected.
_Avoid_: Provider payload, implicit string concatenation

**ResourceLoader**:
The Coding Agent boundary that discovers, scopes, validates, and trust-filters resources before constructing AgentResources.
_Avoid_: AgentResources, Extension loader

**Project Trust**:
A Coding Agent input-loading decision that permits behavior-changing project resources to be discovered from a canonical workspace path; it neither restricts Tools nor creates an execution sandbox.
_Avoid_: Workspace Boundary, permission approval

**Context File**:
A conventional project instruction file such as `AGENTS.md` that is loaded as ordinary model context by default and can be disabled explicitly without being treated as executable Extension code.
_Avoid_: Extension, trusted settings

**Resource Scope**:
The origin and precedence of a discovered Coding Agent resource: explicit, project, user, or built-in.
_Avoid_: Python package scope, Session branch

**Interface Adapter**:
A user-facing shell that translates input into AgentSession commands and renders Session state and Events without owning agent behavior.
_Avoid_: AgentRuntime, ModelAdapter

**RunResult**:
The versioned public result of an Agent Run, containing its identity, terminal status, final content, usage, and diagnostic references independently of terminal rendering.
_Avoid_: AssistantMessage, process stdout

**Event Envelope**:
The versioned JSONL representation of one ordered Event, carrying Session and Run correlation, sequence, type, and payload for machine consumers.
_Avoid_: Run Trace record, terminal log line

**Event**:
A passive record of something that occurred during harness execution and cannot alter that execution; high-level subscribers are awaited in registration order to provide lifecycle barriers.
_Avoid_: Hook, command

**Event Barrier**:
A lifecycle point, notably `message_end` or `agent_end`, whose high-level Event subscribers must settle in registration order before the Runtime crosses it, without allowing their return values to alter agent behavior.
_Avoid_: Hook, queue backpressure protocol

**Observational Event Stream**:
The low-level ordered asynchronous Event iterator exposed through an AgentRunHandle for interfaces and integrations, without making consumer work a Runtime barrier.
_Avoid_: High-level subscriber, Hook

**Hook**:
An ordered extension point that may inspect, transform, replace, or block a pending harness action.
_Avoid_: Event, listener

**Session**:
One tree-structured agent conversation together with the durable state needed to continue its settled history.
_Avoid_: Message list, Agent Run

**SessionStore**:
The persistence boundary through which Sessions are created, retrieved, and updated without exposing a storage format to the Runtime.
_Avoid_: Session, JSONL file

**Workspace Identity**:
The stable identifier derived from a canonical absolute workspace path and used to group local Sessions and Runs without writing harness state into the project.
_Avoid_: Workspace display name, Session ID

**Settled Boundary**:
A point where the complete outcome of a message, model response, or tool execution has been accepted into the Session and is safe to resume from.
_Avoid_: Stream checkpoint, exactly-once boundary

**Compaction**:
A Session operation that replaces an older active-branch prefix in model context with a durable summary while preserving the original conversation tree.
_Avoid_: Message deletion, output truncation

**CompactionPolicy**:
The explicit settings and cut-point rules that reserve model response capacity, retain a recent valid message suffix, and trigger threshold, manual, or overflow Compaction.
_Avoid_: Token estimator, destructive history pruning

**Retained Tail**:
The materialized, structurally valid recent ModelMessage suffix stored with a Compaction entry and used directly when rebuilding model context.
_Avoid_: Pointer-only cut point, arbitrary last messages

**AgentRuntime**:
The in-memory execution engine that advances an agent conversation through model turns, tool calls, queues, Events, and Hooks.
_Avoid_: AgentSession, Coding Agent

**AgentSession**:
The durable application-facing agent conversation that coordinates an AgentRuntime with its Session and lifecycle services.
_Avoid_: AgentRuntime, SessionStore

**Ephemeral Session**:
An AgentSession backed only by in-memory Session and observability stores, created explicitly when no Session, Trace, Annotation, or Artifact may be written to disk.
_Avoid_: Unsaved crash state, temporary JSONL file

**Agent Run**:
One accepted unit of work from an initial prompt through all resulting model turns and tool calls until completion, failure, or abort.
_Avoid_: Session, model turn

**Run Snapshot**:
The immutable effective model, generation, Tool, Extension, Hook, resource, and policy configuration captured when an Agent Run starts and identified in its Run Trace.
_Avoid_: Mutable Session settings, provider request body

**RunGuard**:
An optional settled-turn predicate that can stop an Agent Run according to caller-supplied limits such as turns, Tool calls, elapsed time, or total usage, without imposing aggregate limits by default.
_Avoid_: Provider output limit, mandatory Runtime budget

**AgentRunHandle**:
The live handle returned when an Agent Run starts, exposing its identifier, passive Event stream, cancellation, and eventual result without owning Session input queues.
_Avoid_: AgentSession, background task

**Cancellation Settlement**:
The coordinated terminal transition that propagates cancellation, preserves completed outcomes, materializes cancelled pending Tool results, persists an aborted assistant outcome, and returns unconsumed queued input.
_Avoid_: Immediate process exit, silent task cancellation

**Terminating ToolResult**:
A ToolResult carrying `terminate=True`; automatic model continuation is skipped only when every finalized result in its Tool Batch carries that hint.
_Avoid_: Model Stop Reason, process exit

**Run Trace**:
A versioned, append-only diagnostic record correlated with one Agent Run, capturing timings, attempts, usage, tool activity, compaction, cancellation, and failures without acting as resumable Session state.
_Avoid_: Session history, recovery log

**TraceSink**:
The replaceable observation boundary that accepts Run Trace records as execution occurs, independently of SessionStore commits.
_Avoid_: Event listener with business behavior, SessionStore

**RunStore**:
The queryable persistence boundary for versioned Run Traces and appended outcome annotations, supporting inspection and export without interpreting terminal output.
_Avoid_: SessionStore, optimization engine

**Session Lock**:
The cross-process exclusive writer lease held while a durable Session is being interactively controlled or advanced by an Agent Run, without preventing read-only inspection.
_Avoid_: Process-global lock, workspace lock

**Run Annotation**:
An append-only evaluation record attached to an Agent Run, such as outcome, score, tags, or notes, without rewriting its original Run Trace.
_Avoid_: Trace mutation, assistant message

**Run Artifact**:
A local content object referenced from a Run Trace when a payload is too large or unsuitable for inline JSONL storage, identified by media type, size, and content hash.
_Avoid_: Inline event payload, uploaded telemetry

**Trace Redactor**:
The replaceable pre-persistence filter that removes credentials, authorization material, and configured sensitive fields from Run Trace records and Run Artifacts without modifying Session execution state.
_Avoid_: Session compaction, permission hook

**Schema Version**:
The independent persisted-format compatibility marker carried by Session, Trace, Annotation, Artifact-reference, and Event-envelope records.
_Avoid_: Python package version, Checkpoint number

**Release Gate**:
The complete cross-platform, offline-first verification contract that all Chapter Notebooks, Checkpoint Packages, generated production sources, persistence contracts, and critical failure paths must satisfy before version one is published.
_Avoid_: Chapter-local test, optional online smoke test

**Steering Message**:
A queued user message intended to redirect the active Agent Run at its next turn boundary.
_Avoid_: Follow-up Message, ordinary prompt

**Follow-up Message**:
A queued user message intended to begin work only after the active Agent Run settles.
_Avoid_: Steering Message, ordinary prompt

**Raw Notebook**:
The experimental record that preserves the original evolution of the harness without serving as a course or infrastructure authority.
_Avoid_: Course source, production source
