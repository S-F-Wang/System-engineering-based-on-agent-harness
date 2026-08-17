# Agent Harness — Chapter 6 Checkpoint

This cumulative Checkpoint adds Compaction as an `AgentSession` operation available only at a Settled Boundary. Manual Compaction accepts focus instructions; threshold Compaction uses adaptive reserve and retained-token budgets; normalized context overflow gets one separate recovery attempt rather than entering ordinary transient retry.

`CompactionStrategy.plan(...)` selects a materialized Retained Tail without splitting an assistant Tool Call from its complete `ToolResultMessage` batch. `CompactionPolicy.resolve(...)` applies the small- and broad-context formulas from ADR 0048. Token estimation remains an explicit replaceable seam and carries no claim of Provider accounting.

Each durable `CompactionCheckpoint` stores a versioned structured summary, estimated tokens before Compaction, Provider summary usage when available, its trigger, and the complete retained messages. Session entries remain append-only: `history()` navigates the original conversation, while `effective_history()` materializes the latest summary, Retained Tail, and later settlements for the next model request.

Summary generation uses the Runtime's configured `ModelAdapter`, `ModelSpec`, `RetryPolicy`, sleeper, and cancellation path with `ModelOperation.COMPACTION`. Threshold failure records a warning and preserves history; failed overflow recovery terminates clearly with `compaction_failed`. An unknown context window disables threshold triggering with one warning while leaving overflow recovery available.

Both `MemorySessionStore` and `JSONLSessionStore` persist the same checkpoint contract. JSONL schema minor 1 adds Compaction records while continuing to read prior major-1 Session files.
