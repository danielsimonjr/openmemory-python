"""Security tests for openmemory.embeddings model loading.

These tests assert that we never load Hugging Face models with
``trust_remote_code=True`` and that we always pin a revision so an
upstream tampered/reuploaded weight cannot silently swap in. They also
verify the ``OPENMEMORY_EMBEDDING_MODEL`` / ``OPENMEMORY_EMBEDDING_REVISION``
env-var overrides.

The tests stub :class:`sentence_transformers.SentenceTransformer` with a
recorder so they run offline and don't require any network or HF cache.
"""
from __future__ import annotations

import importlib
import os
from unittest import mock

import pytest

from openmemory import embeddings as emb_mod
from openmemory.embeddings import (
    DEFAULT_MODELS,
    DEFAULT_REVISIONS,
    EmbeddingProvider,
    _resolve_default_models,
    _resolve_revision,
)
from openmemory.types import SectorType


class _StubSentenceTransformer:
    """Records every constructor call so tests can assert on the kwargs."""

    calls: list[dict] = []

    def __init__(self, model_name_or_path, **kwargs):
        type(self).calls.append({"model": model_name_or_path, **kwargs})
        # Minimal surface so callers can use .encode() without crashing if
        # they want to extend tests; embeddings_security tests do not.
        self.model_name = model_name_or_path

    def encode(self, *args, **kwargs):  # pragma: no cover - not exercised here
        import numpy as np

        return np.zeros(384, dtype=np.float32)


@pytest.fixture(autouse=True)
def _patch_sentence_transformer(monkeypatch):
    _StubSentenceTransformer.calls = []
    monkeypatch.setattr(emb_mod, "SentenceTransformer", _StubSentenceTransformer)
    # Always start from a clean env so per-test overrides are explicit.
    monkeypatch.delenv("OPENMEMORY_EMBEDDING_MODEL", raising=False)
    monkeypatch.delenv("OPENMEMORY_EMBEDDING_REVISION", raising=False)
    yield


# ---------------------------------------------------------------------------
# Hardened defaults
# ---------------------------------------------------------------------------

def test_default_load_disables_trust_remote_code():
    """The very first model load must pass trust_remote_code=False."""
    provider = EmbeddingProvider()
    provider._get_model(DEFAULT_MODELS[SectorType.SEMANTIC])
    assert _StubSentenceTransformer.calls, "model never constructed"
    call = _StubSentenceTransformer.calls[0]
    assert call["trust_remote_code"] is False


def test_default_load_pins_revision():
    """Default models must load at the pinned revision, not at HEAD."""
    provider = EmbeddingProvider()
    name = DEFAULT_MODELS[SectorType.SEMANTIC]
    provider._get_model(name)
    call = _StubSentenceTransformer.calls[0]
    assert call.get("revision") == DEFAULT_REVISIONS[name]


def test_default_load_prefers_safetensors():
    """Hardened path requests safetensors to dodge pickle RCE."""
    provider = EmbeddingProvider()
    provider._get_model(DEFAULT_MODELS[SectorType.SEMANTIC])
    call = _StubSentenceTransformer.calls[0]
    assert call.get("model_kwargs") == {"use_safetensors": True}


def test_every_sector_default_is_pinned():
    """No default model should ever ship without a pinned revision."""
    for sector, model_name in DEFAULT_MODELS.items():
        assert model_name in DEFAULT_REVISIONS, (
            f"sector {sector!r} default model {model_name!r} "
            "has no pinned revision in DEFAULT_REVISIONS"
        )


# ---------------------------------------------------------------------------
# Env-var overrides
# ---------------------------------------------------------------------------

def test_env_model_override_applies_to_all_sectors(monkeypatch):
    monkeypatch.setenv("OPENMEMORY_EMBEDDING_MODEL", "my/custom-embedding")
    resolved = _resolve_default_models()
    assert set(resolved.values()) == {"my/custom-embedding"}
    assert set(resolved.keys()) == set(DEFAULT_MODELS.keys())


def test_env_revision_override(monkeypatch):
    monkeypatch.setenv("OPENMEMORY_EMBEDDING_REVISION", "deadbeef" * 5)
    # Override applies regardless of model name, including unknown ones.
    assert _resolve_revision("some/model") == "deadbeef" * 5
    assert _resolve_revision("all-MiniLM-L6-v2") == "deadbeef" * 5


def test_env_overrides_propagate_to_load(monkeypatch):
    monkeypatch.setenv("OPENMEMORY_EMBEDDING_MODEL", "my/custom-embedding")
    monkeypatch.setenv("OPENMEMORY_EMBEDDING_REVISION", "abc123" * 5)

    provider = EmbeddingProvider()
    # The provider should now have picked up the env-var model for every sector.
    assert all(name == "my/custom-embedding" for name in provider.models.values())

    provider._get_model("my/custom-embedding")
    call = _StubSentenceTransformer.calls[0]
    assert call["model"] == "my/custom-embedding"
    assert call["trust_remote_code"] is False
    assert call["revision"] == "abc123" * 5


# ---------------------------------------------------------------------------
# Safetensors fallback
# ---------------------------------------------------------------------------

def test_safetensors_fallback_keeps_trust_remote_code_off(monkeypatch):
    """If safetensors weights aren't available, we still must not enable
    trust_remote_code or float off the pinned revision."""
    attempts: list[dict] = []

    class _FlakySentenceTransformer:
        def __init__(self, model_name_or_path, **kwargs):
            attempts.append({"model": model_name_or_path, **kwargs})
            if kwargs.get("model_kwargs", {}).get("use_safetensors"):
                raise OSError("no safetensors weights for this repo")
            self.model_name = model_name_or_path

    monkeypatch.setattr(emb_mod, "SentenceTransformer", _FlakySentenceTransformer)

    provider = EmbeddingProvider()
    provider._get_model("all-MiniLM-L6-v2")

    # Two attempts: first with safetensors, then fallback.
    assert len(attempts) == 2
    first, second = attempts
    assert first["model_kwargs"] == {"use_safetensors": True}
    assert second["model_kwargs"] == {"use_safetensors": False}
    # Critically, neither attempt may relax trust_remote_code or drop the pin.
    for attempt in attempts:
        assert attempt["trust_remote_code"] is False
        assert attempt.get("revision") == DEFAULT_REVISIONS["all-MiniLM-L6-v2"]
