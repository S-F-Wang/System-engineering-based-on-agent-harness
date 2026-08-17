# Version-one verification report

Date: 2026-08-17

## Decision

The version-one artifact chain passes the complete Release Gate locally and on
all three required GitHub Actions platforms. Candidate commit
`e4fc6e930a51033c964cfbe8e3da71cf2af02b4e` is green on Linux, Windows, and
macOS in [workflow run 32018267396](https://github.com/S-F-Wang/System-engineering-based-on-agent-harness/actions/runs/32018267396).
Version one is release-ready. This report does not publish, tag, or otherwise
create a release.

The authoritative chain is:

`course/notebooks/01..09` → `course/checkpoints/ch01..ch09` →
`src/agent_harness`

The historical `notebooks/lessons/` and `notebooks/raw/` trees are retained
unchanged as non-authoritative historical evidence. They are not package,
Checkpoint, notebook, or production-generation inputs.

## Exact verification commands

The local candidate was checked with Python 3.11.15 and uv 0.11.24:

```bash
.venv/bin/python -m pytest -q tests/test_release_gate.py
.venv/bin/mypy src/agent_harness course/tools
PIP_NO_INDEX=1 AGENT_HARNESS_REAL_SMOKE=0 OPENAI_API_KEY= \
  .venv/bin/python -m course.tools.release
PYTHONPATH=course/checkpoints/ch09/src PIP_NO_INDEX=1 \
  .venv/bin/python -m pytest -q -p no:cacheprovider \
  course/checkpoints/ch09/tests
.venv/bin/python -m pytest -q
uv build --offline
.venv/bin/python -m pip install --no-deps --no-build-isolation \
  --force-reinstall .
.venv/bin/python -c "import agent_harness; print(agent_harness.__name__)"
```

The complete gate returned all nine Chapters with
`compile/install/import/tests` green and deterministic production digest
`c9ee419ee5dd112ec04ac53f25e2281f08a6ce9685ab68f82506efb11cf2d708`.

Observed local results:

- Release tooling: 5 passed.
- Root course suite: 36 passed.
- Final cumulative Checkpoint suite: 146 passed, 1 credential-gated smoke
  skipped.
- Mypy: no issues in 18 source files.
- Source distribution, wheel, forced offline installation, and import: pass.

## Platform matrix

| Platform | Python | Result for this candidate | Source of truth |
| --- | --- | --- | --- |
| Linux | 3.11 | Pass | Workflow run 32018267396; 3m04s |
| Windows | 3.11 | Pass | Workflow run 32018267396; 5m03s |
| macOS | 3.11 | Pass | Workflow run 32018267396; 14m20s |

The workflow first materializes `uv.lock`, then sets `UV_OFFLINE=1`,
`PIP_NO_INDEX=1`, disables the real smoke opt-in, and runs the same release
command on `ubuntu-latest`, `windows-latest`, and `macos-latest`. Every matrix
job ran the release-tool tests, complete offline Release Gate, platform type
check, and offline distribution build/install successfully. No platform result
was inferred from the local host.

## Release evidence

Chapter 9's cumulative suite is the executable release suite. The Release Gate
checks that each named test still exists before running the Checkpoint:

| Requirement | Evidence |
| --- | --- |
| Classified retries | `test_runtime.py::test_transient_failures_retry_on_the_default_two_four_eight_schedule`; `test_runtime.py::test_runtime_classifies_retryability_from_normalized_error_codes` |
| Partial-stream failure | `test_runtime.py::test_provider_failure_after_acceptance_settles_partial_assistant_outcome` |
| Cancellation settlement | `test_run_control.py::test_cancellation_settles_parallel_tools_and_returns_unconsumed_input` |
| Session Lock contention | `test_durable_sessions.py::test_session_writer_lease_contends_across_processes` |
| Path and link escape attempts | `test_coding_tools.py::test_file_tools_reject_parent_traversal_outside_the_workspace`; `test_coding_tools.py::test_file_tools_reject_symlink_escape_and_unsafe_nonexistent_parent` |
| Atomic text mutation | `test_coding_tools.py::test_text_edit_commits_atomically_without_losing_file_permissions` |
| Settled-boundary crash recovery | `test_durable_sessions.py::test_incomplete_final_jsonl_record_is_reported_and_not_replayed`; `test_durable_sessions.py::test_continuation_discards_an_uncommitted_tail_before_appending` |
| Compaction | `test_compaction.py::test_manual_compaction_persists_a_checkpoint_and_changes_only_model_context` plus threshold, overflow, cancellation, and JSONL recovery cases in the same module |
| JSONL and output modes | `test_cli.py::test_exec_json_uses_flag_precedence_and_persists_by_default`; `test_cli.py::test_exec_jsonl_emits_ordered_events_and_mandatory_run_end`; `test_cli.py::test_chat_runs_a_scripted_multi_turn_terminal_session` |
| Trace redaction | `test_observability.py::test_artifact_redaction_preserves_full_sanitized_output` |
| Persisted versions and nondestructive migration | `test_durable_sessions.py::test_session_schema_accepts_optional_fields_and_rejects_unknown_major`; `test_durable_sessions.py::test_migration_validates_a_new_file_and_preserves_the_original`; equivalent Run Trace coverage in `test_observability.py` |

The supplementary real OpenAI-compatible smoke test remains skipped unless
`AGENT_HARNESS_REAL_SMOKE=1` and all three explicit endpoint, API-key, and model
variables are supplied. It is never evidence in place of deterministic tests.

## Frozen scope and dependency boundary

The reusable distribution has exactly three runtime dependencies:
`jsonschema`, `openai`, and `platformdirs`. Notebook, test, build, and type-check
dependencies remain authoring-only.

Version one excludes a full TUI, RPC/server mode, MCP, subagents, multimodal
content, an optimizer, remote Extension installation, a model catalog, and a
built-in sandbox. The shipped Bash tool uses host authority within its fixed
workspace; it is not a sandbox.

## Known limitations

- Only text and Tool-call Content Blocks are supported.
- The sole production ModelAdapter targets OpenAI-compatible streaming Chat
  Completions; provider quirks outside that contract require an adapter change.
- Bash requires a real Bash installation, including Git Bash on Windows.
- Persistence is local and single-writer per Session; there is no remote or
  distributed coordination.
- The deterministic suite proves contract behavior with scripted and loopback
  adapters. Real endpoint availability and provider behavior remain
  supplementary operational checks.
