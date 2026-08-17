# Support only OpenAI-compatible providers in version one

Version one will ship one production ModelAdapter for OpenAI-compatible endpoints configured by base URL, credentials, model, and explicit compatibility options; it will not implement native provider SDKs, OAuth, subscription login, or dynamic model catalogs. A ScriptedModelAdapter remains part of the offline course and test infrastructure but is not advertised as a production Provider. This keeps the chapters focused on harness engineering while preserving deterministic conformance and regression tests.
