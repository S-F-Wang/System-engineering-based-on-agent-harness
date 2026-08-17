# Separate agent, model, and provider messages

The Reusable Harness will persist and operate on typed AgentMessages, transform them into the narrower provider-neutral ModelMessage set before inference, and confine provider wire formats to ModelAdapters. Content will use explicit blocks such as text, thinking, image, and tool call rather than unconstrained dictionaries. This adds an early modeling step to the course but prevents session, extension, and agent semantics from becoming coupled to one provider SDK.
