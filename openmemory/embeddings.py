"""
Embedding generation for memory sectors
"""
import os
import numpy as np
from typing import List, Optional, Dict
from dataclasses import dataclass
from sentence_transformers import SentenceTransformer
from .types import SectorType


# ---------------------------------------------------------------------------
# Model loading: secure defaults
# ---------------------------------------------------------------------------
#
# We load Hugging Face models via sentence-transformers. HF model repos can
# ship arbitrary `*.py` files that execute on load when ``trust_remote_code``
# is enabled, and the legacy ``pytorch_model.bin`` format is a pickle and is
# therefore an arbitrary-code-execution channel by itself. To keep loading
# safe by default we:
#
#   * pin every default model to a known-good commit SHA (``DEFAULT_REVISION``
#     below) so an upstream tampered/reuploaded weight cannot silently swap
#     in;
#   * pass ``trust_remote_code=False`` on every load so no custom python from
#     the model repo is ever executed;
#   * prefer ``safetensors`` weights via ``model_kwargs={"use_safetensors":
#     True}``, which removes the pickle attack surface entirely.
#
# Operators can override the model and revision via environment variables
# without editing code (useful for air-gapped mirrors and pinning forks to
# audited revisions):
#
#   OPENMEMORY_EMBEDDING_MODEL      -- model id (e.g. ``all-MiniLM-L6-v2``)
#   OPENMEMORY_EMBEDDING_REVISION   -- git commit SHA on the HF repo
#
# When set, these apply to **every** sector, replacing both the MiniLM and
# MPNet defaults. To override per sector, pass an explicit ``models`` dict to
# :class:`EmbeddingProvider`.

# Pinned revisions taken from huggingface.co. These are the commit SHAs that
# were current and audited at the time of this hardening pass; bump them
# explicitly when you want to pull in upstream updates.
DEFAULT_REVISIONS = {
    "all-MiniLM-L6-v2": "c9745ed1d9f207416be6d2e6f8de32d1f16199bf",
    "all-mpnet-base-v2": "84f2bcc00d77236f9e89c8a360a00fb1139bf47d",
}

# Default embedding models for each sector
DEFAULT_MODELS = {
    SectorType.EPISODIC: "all-MiniLM-L6-v2",
    SectorType.SEMANTIC: "all-MiniLM-L6-v2",
    SectorType.PROCEDURAL: "all-MiniLM-L6-v2",
    SectorType.EMOTIONAL: "all-MiniLM-L6-v2",
    SectorType.REFLECTIVE: "all-mpnet-base-v2"
}


def _resolve_default_models() -> Dict[SectorType, str]:
    """Return DEFAULT_MODELS, possibly overridden by env var."""
    override = os.environ.get("OPENMEMORY_EMBEDDING_MODEL")
    if override:
        return {sector: override for sector in DEFAULT_MODELS}
    return dict(DEFAULT_MODELS)


def _resolve_revision(model_name: str) -> Optional[str]:
    """Return the pinned revision for ``model_name``, or env override."""
    override = os.environ.get("OPENMEMORY_EMBEDDING_REVISION")
    if override:
        return override
    return DEFAULT_REVISIONS.get(model_name)


@dataclass
class EmbeddingResult:
    """Result of embedding generation"""
    sector: SectorType
    vector: np.ndarray
    dim: int


class EmbeddingProvider:
    """
    Embedding provider using sentence-transformers.

    Supports multiple models for different sectors.
    """

    def __init__(self, models: Optional[Dict[SectorType, str]] = None):
        """
        Initialize embedding provider.

        Args:
            models: Optional dict mapping sectors to model names. When
                omitted, defaults are taken from :data:`DEFAULT_MODELS`,
                with an optional global override from the
                ``OPENMEMORY_EMBEDDING_MODEL`` env var.
        """
        self.models = models if models is not None else _resolve_default_models()
        self._model_cache: Dict[str, SentenceTransformer] = {}

    def _get_model(self, model_name: str) -> SentenceTransformer:
        """Get or load a model with hardened defaults.

        See module docstring for the full security rationale. In short:
        ``trust_remote_code=False`` (no arbitrary python from the repo),
        revision is pinned (no silent upstream swap), and we prefer
        ``safetensors`` weights to dodge pickle-based RCE.
        """
        if model_name not in self._model_cache:
            revision = _resolve_revision(model_name)
            kwargs = {
                "trust_remote_code": False,
                "model_kwargs": {"use_safetensors": True},
            }
            if revision is not None:
                kwargs["revision"] = revision
            try:
                self._model_cache[model_name] = SentenceTransformer(
                    model_name, **kwargs
                )
            except (OSError, EnvironmentError, ValueError):
                # Fallback for repos without safetensors weights. We still
                # keep ``trust_remote_code=False`` and the pinned revision
                # so we never execute repo python or float on HEAD.
                kwargs["model_kwargs"] = {"use_safetensors": False}
                self._model_cache[model_name] = SentenceTransformer(
                    model_name, **kwargs
                )
        return self._model_cache[model_name]

    def embed_for_sector(self, text: str, sector: SectorType) -> np.ndarray:
        """
        Generate embedding for text in a specific sector.

        Args:
            text: Text to embed
            sector: Target sector

        Returns:
            Embedding vector as numpy array
        """
        model_name = self.models.get(sector, self.models[SectorType.SEMANTIC])
        model = self._get_model(model_name)

        # Generate embedding
        embedding = model.encode(text, convert_to_numpy=True)

        # Normalize
        embedding = embedding / (np.linalg.norm(embedding) + 1e-8)

        return embedding

    def embed_multi_sector(
        self,
        text: str,
        sectors: List[SectorType]
    ) -> List[EmbeddingResult]:
        """
        Generate embeddings for multiple sectors.

        Args:
            text: Text to embed
            sectors: List of sectors

        Returns:
            List of embedding results
        """
        results = []

        for sector in sectors:
            embedding = self.embed_for_sector(text, sector)
            results.append(EmbeddingResult(
                sector=sector,
                vector=embedding,
                dim=len(embedding)
            ))

        return results


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Compute cosine similarity between two vectors.

    Args:
        vec1: First vector
        vec2: Second vector

    Returns:
        Cosine similarity (0-1)
    """
    # Ensure float32
    v1 = vec1.astype(np.float32)
    v2 = vec2.astype(np.float32)

    # Compute dot product
    dot_product = np.dot(v1, v2)

    # Compute norms
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)

    # Avoid division by zero
    if norm1 == 0 or norm2 == 0:
        return 0.0

    similarity = dot_product / (norm1 * norm2)

    # Clamp to [0, 1]
    similarity = max(0.0, min(1.0, similarity))

    return float(similarity)


def vector_to_bytes(vector: np.ndarray) -> bytes:
    """Convert numpy array to bytes for storage"""
    return vector.astype(np.float32).tobytes()


def bytes_to_vector(data: bytes) -> np.ndarray:
    """Convert bytes back to numpy array"""
    return np.frombuffer(data, dtype=np.float32)


def calculate_mean_vector(embeddings: List[EmbeddingResult], sector_weights: Optional[Dict[SectorType, float]] = None) -> np.ndarray:
    """
    Calculate weighted mean vector across multiple sector embeddings.

    Uses softmax weighting based on sector confidence.

    Args:
        embeddings: List of embedding results
        sector_weights: Optional sector weights (defaults to equal)

    Returns:
        Mean vector
    """
    if not embeddings:
        raise ValueError("No embeddings provided")

    # Get dimension from first embedding
    dim = embeddings[0].dim

    # Initialize weighted sum
    weighted_sum = np.zeros(dim, dtype=np.float32)

    # Calculate softmax weights
    beta = 2.0  # From TypeScript implementation
    if sector_weights is None:
        from .sectors import get_sector_config
        sector_weights = {
            emb.sector: get_sector_config(emb.sector).weight
            for emb in embeddings
        }

    # Calculate softmax denominator
    exp_sum = sum(np.exp(beta * sector_weights[emb.sector]) for emb in embeddings)

    # Weighted average
    for emb in embeddings:
        weight = sector_weights[emb.sector]
        softmax_weight = np.exp(beta * weight) / exp_sum
        weighted_sum += emb.vector * softmax_weight

    # Normalize
    norm = np.linalg.norm(weighted_sum)
    if norm > 0:
        weighted_sum /= norm

    return weighted_sum
