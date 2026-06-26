# Changelog

All notable changes to this project are documented here.

## [Unreleased]

### Fixed
- **Thread-safe SQLite access (`openmemory/storage.py`).** DB operations used `asyncio.to_thread`, which dispatches to the shared default thread pool — concurrent coroutines could therefore touch the single `sqlite3.Connection` from different worker threads at once, which a sqlite connection does not support (corruption / "recursive use of cursors" risk). All DB calls now run through a dedicated single-worker `ThreadPoolExecutor` (`Storage._run`), serializing access onto one thread. Verified with an insert/get round-trip plus 20 concurrent reads.
- **Observability of tool-call failures (`openmemory/mcp/server.py`).** The catch-all handler returned `{"error": str(e)}` to the client but logged nothing server-side; it now prints the full traceback to stderr (stdout is the JSON-RPC channel) before returning the error.

### Security

- **Hardened Hugging Face embedding model loading.** `EmbeddingProvider`
  now loads every model with `trust_remote_code=False`, prefers
  `safetensors` weights via `model_kwargs={"use_safetensors": True}`,
  and pins each default model to an audited commit SHA in
  `DEFAULT_REVISIONS`. This closes the pickled-weights RCE channel and
  the "upstream silently reuploads weights" channel.
- Added `OPENMEMORY_EMBEDDING_MODEL` and `OPENMEMORY_EMBEDDING_REVISION`
  env-var overrides for air-gapped mirrors and audited forks.
- Added `tests/test_embeddings_security.py` asserting the hardened
  defaults (8 tests, all offline via a stubbed `SentenceTransformer`).
- README documents the security posture and the override env vars.
