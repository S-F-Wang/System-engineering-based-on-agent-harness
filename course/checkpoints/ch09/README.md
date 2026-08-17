# Agent Harness — Chapter 9 / Version 1 Checkpoint

This final cumulative Checkpoint freezes the reusable Python interface and ships
the `omega` adapters over the same `AgentSession` contract. `create_session()`
builds a general Session; `create_coding_session()` explicitly installs local
resources plus `read`, `write`, `edit`, and a real host `bash`. Callers may use
completion-oriented `await session.run(...)` or controllable
`session.start(...)` with Events, snapshots, Steering, Follow-up, commands, and
coordinated cancellation.

The Bash adapter remains a real host process. Neither Bash nor Extensions are a sandbox guarantee;
the Workspace Boundary confines file adapters but is not a built-in process sandbox.

The library accepts a `ModelAdapter` or explicit `OpenAICompatibleConfig` plus
`ModelSpec`; it never reads environment or dotenv state. The CLI alone translates
`OPENAI_API_KEY`, `OPENAI_BASE_URL`, and `OPENAI_MODEL`. `--model` and
`--base-url` override their non-secret environment equivalents. There is no
API-key flag.

```bash
omega exec "inspect this workspace"
omega exec "machine task" --format json
omega exec "event stream" --format jsonl
omega chat
omega sessions list
omega runs list
```

`omega exec` and `omega chat` create new durable Sessions by default. Continue
only with `--session ID` (or `omega chat --continue`). `--no-save` is the explicit
privacy mode and disables Session, Trace, Annotation, and Artifact persistence as
one boundary. Local records are grouped by canonical Workspace Identity: one
JSONL file per Session and one directory per Run containing append-only Trace and
Annotation JSONL plus content-addressed Artifacts. Nothing is uploaded implicitly.

Text exec mode reserves stdout for final assistant text. JSON emits one versioned
`RunResult`; JSONL emits ordered versioned Event Envelopes and always ends with
`run_end`. Standard Trace records snapshots, platform and available Git evidence,
attempts, retries, latency, usage provenance, Tool activity, errors, cancellation,
and Compaction while redacting credential-shaped fields and bounding previews.
Ordinary Trace failure marks evidence incomplete without changing semantic Run
success; strict tracing is opt-in.

Version one targets Python 3.11+ on Windows, Linux, and macOS. Runtime dependencies
are only `openai`, `jsonschema`, and `platformdirs`, all within the accepted small
cross-platform set. Jupyter, tests, documentation tooling, TUI/RPC/MCP stacks, and
provider catalogs are not runtime dependencies. The deliberate version-one
exclusions are a full TUI, RPC/server mode, MCP, subagents, multimodal content,
optimizer, remote Extension installation, model catalog, and built-in sandbox.
The real OpenAI-compatible smoke test remains explicit, credential-gated, and
supplementary to deterministic offline tests.
