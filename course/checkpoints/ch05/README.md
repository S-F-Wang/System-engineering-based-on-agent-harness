# Agent Harness — Chapter 5 Checkpoint

This cumulative Checkpoint turns `AgentSession` into the durable façade around `AgentRuntime`. `MemorySessionStore` and `JSONLSessionStore` implement the same small `SessionStore` interface: create, read, and acquire one writer. Settlements are append-only tree entries with explicit entry and parent identifiers, so continuing from the active leaf or forking a historical entry never flattens prior history.

`AgentRuntime` exposes a Session-owned settlement sink. Complete user or assistant messages commit independently, while an assistant Tool Call and its ordered `ToolResultMessage` values commit as one structural settlement. A process interruption cannot restore an orphan Tool Call or claim mid-stream or mid-Tool replay.

`JSONLSessionStore` writes one versioned file per Session, flushes every settlement, ignores and reports an incomplete final record, and discards that uncommitted tail under the writer lease before continuation. It accepts unknown optional fields within schema major 1 and rejects unsupported majors with migration guidance. `migrate_session_file` writes and validates a separate current-format file without modifying the original.

`create_agent_session` creates new durable state whenever a Store is supplied without a Session id. Continuation and historical forks are explicit. `no_save=True` creates a fully ephemeral Session and rejects persistence or continuation inputs.

Durable writers use cross-platform operating-system file locks. A competing writer fails promptly with structured `session_busy`, while read-only inspection and independent Sessions remain available.
