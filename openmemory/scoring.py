"""
Hybrid scoring for memory retrieval
"""
import re
from typing import Set
from .decay import sigmoid, boosted_similarity


# Scoring weights (from TypeScript implementation)
SCORING_WEIGHTS = {
    "similarity": 0.6,
    "overlap": 0.2,
    "waypoint": 0.15,
    "recency": 0.05
}


def canonical_token_set(text: str) -> Set[str]:
    """
    Extract canonical token set from text.

    Tokenizes text and normalizes tokens for overlap calculation.

    Args:
        text: Input text

    Returns:
        Set of canonical tokens
    """
    # Convert to lowercase
    text = text.lower()

    # Extract words (alphanumeric + numbers)
    tokens = re.findall(r'\b\w+\b', text)

    # Filter out very short tokens and stopwords
    stopwords = {
        'a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'be',
        'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
        'would', 'should', 'could', 'may', 'might', 'can', 'this', 'that',
        'these', 'those', 'it', 'its'
    }

    filtered = {
        token for token in tokens
        if len(token) > 2 and token not in stopwords
    }

    return filtered


def compute_token_overlap(query_tokens: Set[str], memory_tokens: Set[str]) -> float:
    """
    Compute token overlap score.

    Args:
        query_tokens: Set of query tokens
        memory_tokens: Set of memory tokens

    Returns:
        Overlap score (0-1)
    """
    if len(query_tokens) == 0:
        return 0.0

    intersection = query_tokens & memory_tokens
    overlap = len(intersection) / len(query_tokens)

    return overlap


def compute_hybrid_score(
    similarity: float,
    token_overlap: float,
    waypoint_weight: float,
    recency_score: float
) -> float:
    """
    Compute final hybrid score combining multiple factors.

    Formula:
    score = sigmoid(
        w_sim * boosted(sim) +
        w_overlap * overlap +
        w_waypoint * waypoint +
        w_recency * recency
    )

    Args:
        similarity: Cosine similarity score (0-1)
        token_overlap: Token overlap score (0-1)
        waypoint_weight: Waypoint association weight (0-1)
        recency_score: Recency score (0-1)

    Returns:
        Final hybrid score (0-1)
    """
    # Apply boosted similarity transformation
    sim_prime = boosted_similarity(similarity)

    # Weighted combination
    raw_score = (
        SCORING_WEIGHTS["similarity"] * sim_prime +
        SCORING_WEIGHTS["overlap"] * token_overlap +
        SCORING_WEIGHTS["waypoint"] * waypoint_weight +
        SCORING_WEIGHTS["recency"] * recency_score
    )

    # Apply sigmoid to normalize
    final_score = sigmoid(raw_score)

    return final_score


def compute_simhash(text: str) -> str:
    """
    Compute SimHash for deduplication.

    Args:
        text: Input text

    Returns:
        Hexadecimal hash string
    """
    tokens = canonical_token_set(text)

    # Convert tokens to hashes
    hashes = []
    for token in tokens:
        h = 0
        for char in token:
            h = ((h << 5) - h) + ord(char)
            h = h & 0xFFFFFFFF  # Keep 32-bit
        hashes.append(h)

    # Build 64-bit vector
    vec = [0] * 64

    for h in hashes:
        for i in range(64):
            if h & (1 << i):
                vec[i] += 1
            else:
                vec[i] -= 1

    # Convert to hex string
    hash_str = ""
    for i in range(0, 64, 4):
        nibble = (
            (8 if vec[i] > 0 else 0) +
            (4 if vec[i+1] > 0 else 0) +
            (2 if vec[i+2] > 0 else 0) +
            (1 if vec[i+3] > 0 else 0)
        )
        hash_str += format(nibble, 'x')

    return hash_str


def hamming_distance(hash1: str, hash2: str) -> int:
    """
    Compute Hamming distance between two SimHash strings.

    Args:
        hash1: First hash
        hash2: Second hash

    Returns:
        Hamming distance
    """
    dist = 0
    for c1, c2 in zip(hash1, hash2):
        xor = int(c1, 16) ^ int(c2, 16)
        # Count bits in nibble
        dist += bin(xor).count('1')

    return dist
