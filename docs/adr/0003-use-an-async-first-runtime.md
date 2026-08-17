# Use an async-first runtime

The Reusable Harness will expose one async-first runtime for model streaming, agent execution, tool calls, event delivery, cancellation, queued input, and idle settlement. Synchronous tools may be adapted into that runtime and user-facing synchronous conveniences may wrap it, but the project will not maintain a second synchronous Agent implementation; this accepts slightly more ceremony in early chapters to avoid a later breaking rewrite when concurrency and cancellation are introduced.
