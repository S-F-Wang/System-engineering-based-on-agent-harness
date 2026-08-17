# Agent Harness — Chapter 4 Checkpoint

This cumulative Checkpoint adds `AgentSession`, the application-facing control module around `AgentRuntime`. A Session accepts one active Run, rejects a competing prompt explicitly, injects Steering Messages at the next valid turn boundary, and starts queued Follow-up Messages only after the preceding Run has emitted `agent_end`. Independent Sessions remain concurrently runnable.

`SessionRunHandle` exposes ordered Events, coordinated cancellation, and an eventual `SessionRunResult`. Cancellation propagates through provider streaming and Tool execution. Completed parallel Tool results are retained, unfinished calls receive structured `cancelled` ToolResults in source order, an aborted Assistant outcome closes the history, and unconsumed Steering and Follow-up inputs return as typed `pending_inputs`.

`RunGuard` provides opt-in maximum turns, Tool calls, elapsed time, and total-token limits with distinct terminal statuses. There are no aggregate limits by default. Guard and cancellation exits occur only after coherent outcomes are materialized. `AsyncioProcessOperations` demonstrates cancellation of a real host child process without claiming sandbox isolation.

All ordinary tests are deterministic and offline. The inherited real-endpoint smoke remains supplementary and explicitly credential-gated.
