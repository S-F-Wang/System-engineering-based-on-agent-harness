# Agent Harness — Chapter 3 Checkpoint

This cumulative Checkpoint adds explicit structured Tools to the asynchronous Runtime. A Tool declares its name, description, object-shaped JSON Schema, async behavior, concurrency requirement, and output direction. `AgentRuntime` parses and validates model arguments, exposes a stable `PreparedToolCall` preflight value, delegates through a replaceable `ToolExecutor`, and records typed `ToolResultMessage` values before the next model turn.

Read-only Tool batches run concurrently while completion Events reflect actual completion order and conversation history remains in model call order. One sequential or mutating Tool makes the complete batch deterministic and sequential. Invalid JSON, invalid arguments, unknown Tools, and ordinary execution exceptions become safe actionable Tool Errors.

Model-facing output defaults to 50 KB or 2000 lines and carries a structured truncation notice plus a complete-output reference contract when bounded. Automatic continuation stops only when every result in a Tool Batch has `terminate=True`. The core registers no Tool, approval UI, or permission policy implicitly.

All ordinary tests are deterministic and offline. The inherited real-endpoint smoke remains supplementary and explicitly credential-gated.
