# Agent Harness — Chapter 8 Checkpoint

This cumulative Checkpoint adds the optional Coding Agent product layer. `ResourceLoader` resolves only explicit and local project, user, and built-in resources; Project Trust is recorded against a canonical workspace; `PromptAssembler` keeps system context deterministic and Skill instructions progressively disclosed.

`create_coding_tool_preset()` explicitly installs `read`, `write`, `edit`, and `bash`. File Tools enforce a canonical Workspace Boundary, reject binary mutation, and atomically replace text. Bash is a real host process with a fixed working directory, sensitive-environment filtering, cancellation, timeout, bounded model output through the Runtime, and optional sanitized content-addressed Artifacts. Neither Bash nor Extensions are a sandbox guarantee.

Offline tests use fixture Context Files and resources, `ScriptedModelAdapter`, a real local Bash executable, and a loopback OpenAI-compatible endpoint. They do not require a repository-root `AGENTS.md`. The optional real-endpoint smoke test remains skipped unless its documented endpoint, API key, and model variables are deliberately configured.
