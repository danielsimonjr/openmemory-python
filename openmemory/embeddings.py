"""
Embedding generation for memory sectors
"""
import numpy as np
from typing import List, Optional, Dict
from dataclasses import dataclass
from sentence_transformers import SentenceTransformer
from .types import SectorType


# Default embedding models for each sector
DEFAULT_MODELS = {
    SectorType.EPISODIC: "all-MiniLM-L6-v2",
    SectorType.SEMANTIC: "all-MiniLM-L6-v2",
    SectorType.PROCEDURAL: "all-MiniLM-L6-v2",
    SectorType.EMOTIONAL: "all-MiniLM-L6-v2",
    SectorType.REFLECTIVE: "all-mpnet-base-v2"
}


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
            models: Optional dict mapping sectors to model names
        """
        self.models = models or DEFAULT_MODELS
        self._model_cache: Dict[str, SentenceTransformer] = {}

    def _get_model(self, model_name: str) -> SentenceTransformer:
        """Get or load a model"""
        if model_name not in self._model_cache:
            self._model_cache[model_name] = SentenceTransformer(model_name)
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
