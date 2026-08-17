# Agent Harness — Chapter 2 Checkpoint

This cumulative Checkpoint adds one async-first `AgentRuntime` to the Chapter 1 model seam. `start(...)` returns an `AgentRunHandle` with an ordered passive Event stream, cancellation, and an eventual typed `AssistantOutcome`; `run(...)` is its completion-oriented convenience.

The Runtime owns classified retries and bounded backoff. Its default retry schedule is 2, 4, and 8 seconds, and a provider `Retry-After` hint is honored only through 60 seconds. Failures and cancellation after acceptance settle into provider-neutral Assistant outcomes; invalid input still fails before acceptance. Failed partial attempts remain diagnostic Events rather than conversation history, and aggregate Run guards are opt-in.

All ordinary tests are deterministic and offline. The inherited real-endpoint smoke test remains supplementary and runs only when `AGENT_HARNESS_REAL_SMOKE=1` and all three explicit endpoint variables are supplied.
