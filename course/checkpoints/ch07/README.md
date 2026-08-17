# Agent Harness — Chapter 7 Checkpoint

This cumulative Checkpoint adds a bounded, explicitly supplied Extension surface. An `Extension` configures only its public `ExtensionAPI`: Tools with explicit replacement, ordered passive Event subscribers, the five declared Hooks, chat commands, versioned namespaced Custom Entries, Run Annotations, and awaited lifecycle handlers. Core code performs no discovery, installation, or implicit project loading.

High-level subscribers are awaited in registration order. `message_end` remains a persistence barrier before dependent work, while `agent_end` settles before `AgentSession.busy` becomes false. Subscriber failures produce correlated `subscriber_failed` observations and do not change semantic results. `AgentRunHandle.events()` and `SessionRunHandle.events()` remain ordered observational streams whose consumers do not delay Runtime progress.

Hooks fail safe at each intervention seam. Run and model-request failures terminate without uncertain model work, Tool preflight failure blocks execution, Tool-result failure substitutes a safe error, and Compaction failure cancels the operation unless normalized overflow makes recovery terminal. Hook order and effective registrations are frozen in the accepted Run Snapshot.

`RunSnapshot` captures the effective ModelSpec, generation settings, Tools, Extension identities, Hook order, retry, Compaction and guard configuration, prompt and resource hashes, plus a stable fingerprint. Events and outcomes retain Run and snapshot correlation. Runtime reconfiguration can therefore affect only later accepted Runs.

Idle Extension reload awaits teardown before constructing replacements. Initialization identifies its explicit source; teardown failure becomes a warning and cannot leave the Session busy. Custom Entries persist in the Session tree without entering model history, while Run Annotations remain a separate in-memory seam for the later observability layer.
