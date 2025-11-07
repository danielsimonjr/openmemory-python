"""
Memory decay system with dual-process exponential decay
"""
import math
import time
from typing import Optional
from .types import SectorType
from .sectors import get_sector_config


# Decay dynamics constants (from TypeScript implementation)
LAMBDA_ONE_FAST_DECAY_RATE = 0.1
LAMBDA_TWO_SLOW_DECAY_RATE = 0.01
THETA_CONSOLIDATION_COEFFICIENT_FOR_LONG_TERM = 0.3
ALPHA_LEARNING_RATE_FOR_RECALL_REINFORCEMENT = 0.08

# Hybrid parameters
HYBRID_PARAMS = {
    "tau": 3,
    "beta": 2,
    "eta": 0.1,
    "gamma": 0.2,
    "alpha_reinforce": 0.08,
    "t_days": 7,
    "t_max_days": 60,
    "tau_hours": 1,
    "epsilon": 1e-8
}

# Reinforcement parameters
REINFORCEMENT = {
    "salience_boost": 0.1,
    "waypoint_boost": 0.05,
    "max_salience": 1.0,
    "max_waypoint_weight": 1.0,
    "prune_threshold": 0.05
}


def calc_decay(
    sector: SectorType,
    init_salience: float,
    days_since: float,
    seg_idx: Optional[int] = None,
    max_seg: Optional[int] = None
) -> float:
    """
    Calculate decayed salience for a memory.

    Uses dual-process exponential decay:
    - Fast decay (system 1): lambda_1 = 0.1
    - Slow decay (system 2/consolidation): lambda_2 = 0.01
    - Sector-specific decay: lambda_s from sector config

    Args:
        sector: Memory sector type
        init_salience: Initial salience value (0-1)
        days_since: Days since last access
        seg_idx: Optional segment index for segment-based decay adjustment
        max_seg: Optional maximum segment count

    Returns:
        Decayed salience value (0-1)
    """
    config = get_sector_config(sector)
    lambda_sector = config.decay_lambda

    # Adjust lambda based on segment position if provided
    if seg_idx is not None and max_seg is not None and max_seg > 0:
        seg_ratio = math.sqrt(seg_idx / max_seg)
        lambda_sector = lambda_sector * (1 - seg_ratio)

    # Dual-process decay
    fast_decay = math.exp(-LAMBDA_ONE_FAST_DECAY_RATE * days_since)
    slow_decay = THETA_CONSOLIDATION_COEFFICIENT_FOR_LONG_TERM * math.exp(
        -LAMBDA_TWO_SLOW_DECAY_RATE * days_since
    )
    dual_process = fast_decay + slow_decay

    # Sector-specific decay
    sector_decay = math.exp(-lambda_sector * days_since)

    # Blend both decay models (60% dual-process, 40% sector-specific)
    blended_decay = dual_process * 0.6 + sector_decay * 0.4

    # Ensure minimum decay factor of 0.7
    decay_factor = max(0.7, blended_decay)

    # Apply decay to salience
    decayed_salience = init_salience * decay_factor

    return max(0.0, decayed_salience)


def calc_recency_score(last_seen_timestamp: int) -> float:
    """
    Calculate recency score based on time since last access.

    Args:
        last_seen_timestamp: Unix timestamp (ms) of last access

    Returns:
        Recency score (0-1)
    """
    now = int(time.time() * 1000)
    days_since = (now - last_seen_timestamp) / (1000 * 60 * 60 * 24)

    t_days = HYBRID_PARAMS["t_days"]
    t_max_days = HYBRID_PARAMS["t_max_days"]

    score = math.exp(-days_since / t_days) * (1 - days_since / t_max_days)
    return max(0.0, min(1.0, score))


def apply_retrieval_reinforcement(memory_id: str, current_salience: float) -> float:
    """
    Apply reinforcement to memory salience upon retrieval.

    Args:
        memory_id: Memory ID (for potential future context-based adjustment)
        current_salience: Current salience value

    Returns:
        Reinforced salience value
    """
    boost = REINFORCEMENT["salience_boost"]
    new_salience = current_salience + boost
    return min(REINFORCEMENT["max_salience"], new_salience)


def calculate_cross_sector_resonance(
    sector_a: SectorType,
    sector_b: SectorType,
    base_similarity: float
) -> float:
    """
    Calculate cross-sector resonance score.

    Memories from different but related sectors can resonate.

    Args:
        sector_a: First sector
        sector_b: Second sector
        base_similarity: Base similarity score

    Returns:
        Adjusted similarity score
    """
    if sector_a == sector_b:
        return base_similarity

    # Cross-sector resonance weights
    resonance_matrix = {
        (SectorType.EPISODIC, SectorType.EMOTIONAL): 1.2,
        (SectorType.EMOTIONAL, SectorType.EPISODIC): 1.2,
        (SectorType.SEMANTIC, SectorType.PROCEDURAL): 1.1,
        (SectorType.PROCEDURAL, SectorType.SEMANTIC): 1.1,
        (SectorType.REFLECTIVE, SectorType.SEMANTIC): 1.15,
        (SectorType.SEMANTIC, SectorType.REFLECTIVE): 1.15,
        (SectorType.REFLECTIVE, SectorType.EPISODIC): 1.1,
        (SectorType.EPISODIC, SectorType.REFLECTIVE): 1.1,
    }

    # Get resonance factor (default to 1.0 for unrelated sectors)
    resonance_factor = resonance_matrix.get((sector_a, sector_b), 1.0)

    return base_similarity * resonance_factor


def sigmoid(x: float) -> float:
    """Standard sigmoid function"""
    return 1.0 / (1.0 + math.exp(-x))


def boosted_similarity(similarity: float) -> float:
    """
    Apply boosted similarity transformation.

    Uses: 1 - exp(-tau * similarity)

    Args:
        similarity: Raw similarity score (0-1)

    Returns:
        Boosted similarity score
    """
    tau = HYBRID_PARAMS["tau"]
    return 1.0 - math.exp(-tau * similarity)
