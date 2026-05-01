# Changelog

All notable changes to this project are documented here.

## [Unreleased]

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
