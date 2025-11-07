"""
Type definitions for OpenMemory
"""
from dataclasses import dataclass
from typing import Optional, List, Dict, Any
from enum import Enum


class SectorType(str, Enum):
    """Memory sector types"""
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    PROCEDURAL = "procedural"
    EMOTIONAL = "emotional"
    REFLECTIVE = "reflective"


@dataclass
class SectorConfig:
    """Configuration for a memory sector"""
    model: str
    decay_lambda: float
    weight: float
    patterns: List[str]  # Will be compiled to regex


@dataclass
class SectorClassification:
    """Result of content classification"""
    primary: SectorType
    additional: List[SectorType]
    confidence: float


@dataclass
class MemoryRow:
    """Memory database row"""
    id: str
    content: str
    primary_sector: str
    tags: Optional[str]
    meta: Optional[str]
    user_id: Optional[str]
    created_at: int
    updated_at: int
    last_seen_at: int
    salience: float
    decay_lambda: float
    version: int
    simhash: Optional[str]
    segment: int


@dataclass
class MemoryResult:
    """Query result"""
    id: str
    content: str
    score: float
    sectors: List[str]
    primary_sector: str
    path: List[str]
    salience: float
    last_seen_at: int


@dataclass
class Waypoint:
    """Memory association waypoint"""
    src_id: str
    dst_id: str
    weight: float
    created_at: int
    updated_at: int


@dataclass
class AddMemoryRequest:
    """Request to add a memory"""
    content: str
    tags: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None
    salience: Optional[float] = None
    decay_lambda: Optional[float] = None
    user_id: Optional[str] = None


@dataclass
class QueryRequest:
    """Request to query memories"""
    query: str
    k: int = 10
    filters: Optional[Dict[str, Any]] = None
    user_id: Optional[str] = None
    min_score: Optional[float] = None
    sector: Optional[str] = None
