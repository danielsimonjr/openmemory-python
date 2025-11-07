"""
Memory sector classification and configuration
"""
import re
from typing import Dict, Any, Optional
from .types import SectorType, SectorConfig, SectorClassification


# Sector configurations matching TypeScript implementation
SECTOR_CONFIGS: Dict[SectorType, SectorConfig] = {
    SectorType.EPISODIC: SectorConfig(
        model="episodic-optimized",
        decay_lambda=0.015,
        weight=1.2,
        patterns=[
            r"\b(today|yesterday|last\s+week|remember\s+when|that\s+time)\b",
            r"\b(I\s+(did|went|saw|met|felt))\b",
            r"\b(at\s+\d+:\d+|on\s+\w+day|in\s+\d{4})\b",
            r"\b(happened|occurred|experience|event|moment)\b"
        ]
    ),
    SectorType.SEMANTIC: SectorConfig(
        model="semantic-optimized",
        decay_lambda=0.005,
        weight=1.0,
        patterns=[
            r"\b(define|definition|meaning|concept|theory)\b",
            r"\b(what\s+is|how\s+does|why\s+do|facts?\s+about)\b",
            r"\b(principle|rule|law|algorithm|method)\b",
            r"\b(knowledge|information|data|research|study)\b"
        ]
    ),
    SectorType.PROCEDURAL: SectorConfig(
        model="procedural-optimized",
        decay_lambda=0.008,
        weight=1.1,
        patterns=[
            r"\b(how\s+to|step\s+by\s+step|procedure|process)\b",
            r"\b(first|then|next|finally|afterwards)\b",
            r"\b(install|configure|setup|run|execute)\b",
            r"\b(tutorial|guide|instructions|manual)\b",
            r"\b(click|press|type|enter|select)\b"
        ]
    ),
    SectorType.EMOTIONAL: SectorConfig(
        model="emotional-optimized",
        decay_lambda=0.020,
        weight=1.3,
        patterns=[
            r"\b(feel|feeling|felt|emotion|mood)\b",
            r"\b(happy|sad|angry|excited|worried|anxious|calm)\b",
            r"\b(love|hate|like|dislike|enjoy|fear)\b",
            r"\b(amazing|terrible|wonderful|awful|fantastic|horrible)\b",
            r"[!]{2,}|[\?\!]{2,}"
        ]
    ),
    SectorType.REFLECTIVE: SectorConfig(
        model="reflective-optimized",
        decay_lambda=0.001,
        weight=0.8,
        patterns=[
            r"\b(think|thinking|thought|reflect|reflection)\b",
            r"\b(realize|understand|insight|conclusion|lesson)\b",
            r"\b(why|purpose|meaning|significance|impact)\b",
            r"\b(philosophy|wisdom|belief|value|principle)\b",
            r"\b(should\s+have|could\s+have|if\s+only|what\s+if)\b"
        ]
    )
}


# Compile regex patterns
COMPILED_PATTERNS: Dict[SectorType, list] = {}
for sector, config in SECTOR_CONFIGS.items():
    COMPILED_PATTERNS[sector] = [
        re.compile(pattern, re.IGNORECASE)
        for pattern in config.patterns
    ]


def classify_content(content: str, metadata: Optional[Dict[str, Any]] = None) -> SectorClassification:
    """
    Classify content into memory sectors using pattern matching.

    Args:
        content: Text content to classify
        metadata: Optional metadata that may contain sector hint

    Returns:
        SectorClassification with primary sector, additional sectors, and confidence
    """
    # Check if sector is specified in metadata
    if metadata and "sector" in metadata:
        sector_str = metadata["sector"]
        try:
            sector = SectorType(sector_str)
            return SectorClassification(
                primary=sector,
                additional=[],
                confidence=1.0
            )
        except ValueError:
            pass  # Invalid sector, continue with pattern matching

    # Score each sector based on pattern matches
    scores: Dict[SectorType, float] = {}

    for sector in SectorType:
        score = 0.0
        config = SECTOR_CONFIGS[sector]
        patterns = COMPILED_PATTERNS[sector]

        for pattern in patterns:
            matches = pattern.findall(content)
            if matches:
                score += len(matches) * config.weight

        scores[sector] = score

    # Sort by score
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    # Primary sector is highest scoring
    primary_sector, primary_score = sorted_scores[0]

    # Additional sectors are those scoring above 30% of primary
    threshold = max(1.0, primary_score * 0.3)
    additional_sectors = [
        sector
        for sector, score in sorted_scores[1:]
        if score > 0 and score >= threshold
    ]

    # Calculate confidence
    secondary_score = sorted_scores[1][1] if len(sorted_scores) > 1 else 0
    confidence = (
        min(1.0, primary_score / (primary_score + secondary_score + 1))
        if primary_score > 0
        else 0.2
    )

    # Default to semantic if no patterns matched
    if primary_score == 0:
        primary_sector = SectorType.SEMANTIC

    return SectorClassification(
        primary=primary_sector,
        additional=additional_sectors,
        confidence=confidence
    )


def get_sector_config(sector: SectorType) -> SectorConfig:
    """Get configuration for a sector"""
    return SECTOR_CONFIGS[sector]
