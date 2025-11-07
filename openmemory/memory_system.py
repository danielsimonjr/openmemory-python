"""
Main Memory System - Integrates all OpenMemory components
"""
import uuid
import time
import asyncio
from typing import List, Optional, Dict, Any
from .types import MemoryResult, SectorType, AddMemoryRequest, QueryRequest
from .sectors import classify_content, get_sector_config
from .scoring import (
    compute_simhash,
    hamming_distance,
    canonical_token_set,
    compute_token_overlap,
    compute_hybrid_score
)
from .decay import (
    calc_decay,
    calc_recency_score,
    apply_retrieval_reinforcement
)
from .embeddings import (
    EmbeddingProvider,
    cosine_similarity,
    calculate_mean_vector
)
from .graph import WaypointGraph
from .storage import Storage


class MemorySystem:
    """
    OpenMemory cognitive memory system.

    Provides add, query, and reinforcement operations with
    multi-sector cognitive architecture.
    """

    def __init__(
        self,
        db_path: str = "openmemory.db",
        embedding_provider: Optional[EmbeddingProvider] = None,
        segment_size: int = 10000
    ):
        """
        Initialize Memory System.

        Args:
            db_path: Path to SQLite database
            embedding_provider: Optional custom embedding provider
            segment_size: Max memories per segment before rotation
        """
        self.storage = Storage(db_path)
        self.embeddings = embedding_provider or EmbeddingProvider()
        self.graph = WaypointGraph(self.storage)
        self.segment_size = segment_size

    async def add_memory(
        self,
        content: str,
        user_id: Optional[str] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        salience: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Add a new memory to the system.

        Args:
            content: Memory content
            user_id: Optional user ID for isolation
            tags: Optional tags
            metadata: Optional metadata
            salience: Optional initial salience (default: auto-calculated)

        Returns:
            Dict with memory ID, sector, and deduplication status
        """
        # Compute simhash for deduplication
        simhash = compute_simhash(content)

        # Check for near-duplicate
        existing = await self.storage.get_memory_by_simhash(simhash)
        if existing:
            hamming_dist = hamming_distance(simhash, existing['simhash'])
            if hamming_dist <= 3:
                # Duplicate found - reinforce existing memory
                now = int(time.time() * 1000)
                boosted_salience = min(1.0, existing['salience'] + 0.15)
                await self.storage.update_memory_seen(
                    existing['id'],
                    now,
                    boosted_salience,
                    now
                )
                return {
                    "id": existing['id'],
                    "primary_sector": existing['primary_sector'],
                    "sectors": [existing['primary_sector']],
                    "deduplicated": True
                }

        # Classify content into sectors
        classification = classify_content(content, metadata)
        all_sectors = [classification.primary] + classification.additional

        # Generate ID and timestamp
        memory_id = str(uuid.uuid4())
        now = int(time.time() * 1000)

        # Calculate initial salience
        if salience is None:
            salience = max(0.0, min(1.0, 0.4 + 0.1 * len(classification.additional)))

        # Get segment
        max_seg = await self.storage.get_max_segment()
        seg_count = await self.storage.get_segment_count(max_seg)
        current_segment = max_seg if seg_count < self.segment_size else max_seg + 1

        # Get decay lambda for primary sector
        sector_config = get_sector_config(classification.primary)
        decay_lambda = sector_config.decay_lambda

        # Insert memory
        await self.storage.insert_memory(
            id=memory_id,
            content=content,
            primary_sector=classification.primary.value,
            salience=salience,
            decay_lambda=decay_lambda,
            created_at=now,
            user_id=user_id,
            tags=",".join(tags) if tags else None,
            meta=str(metadata) if metadata else None,
            simhash=simhash,
            segment=current_segment
        )

        # Generate embeddings for all sectors
        embedding_results = self.embeddings.embed_multi_sector(content, all_sectors)

        # Store embeddings
        for emb_result in embedding_results:
            await self.storage.insert_vector(
                memory_id,
                emb_result.sector.value,
                emb_result.vector
            )

        # Calculate and store mean vector
        mean_vector = calculate_mean_vector(embedding_results)
        await self.storage.update_mean_vector(memory_id, mean_vector)

        # Create similarity waypoint
        await self.graph.create_similarity_waypoint(
            memory_id,
            mean_vector,
            threshold=0.75
        )

        return {
            "id": memory_id,
            "primary_sector": classification.primary.value,
            "sectors": [s.value for s in all_sectors],
            "deduplicated": False
        }

    async def query(
        self,
        query: str,
        k: int = 10,
        user_id: Optional[str] = None,
        sectors: Optional[List[str]] = None,
        min_salience: Optional[float] = None
    ) -> List[MemoryResult]:
        """
        Query memories using hybrid search.

        Args:
            query: Query string
            k: Number of results
            user_id: Optional user filter
            sectors: Optional sector filter
            min_salience: Optional minimum salience filter

        Returns:
            List of memory results sorted by score
        """
        # Classify query
        query_classification = classify_content(query)
        query_sectors = [query_classification.primary] + query_classification.additional

        # Filter sectors if specified
        if sectors:
            sector_enums = [SectorType(s) for s in sectors]
            query_sectors = [s for s in query_sectors if s in sector_enums]

        if not query_sectors:
            query_sectors = [SectorType.SEMANTIC]

        # Generate query embeddings for each sector
        query_embeddings = {}
        for sector in query_sectors:
            query_embeddings[sector] = self.embeddings.embed_for_sector(query, sector)

        # Get canonical tokens
        query_tokens = canonical_token_set(query)

        # Search within each sector
        sector_results: Dict[SectorType, List[tuple]] = {}

        for sector in query_sectors:
            query_vec = query_embeddings[sector]
            sector_vecs = await self.storage.get_vectors_by_sector(sector.value)

            similarities = []
            for vec_data in sector_vecs:
                mem_vec = vec_data['v']
                similarity = cosine_similarity(query_vec, mem_vec)
                similarities.append((vec_data['id'], similarity))

            # Sort by similarity
            similarities.sort(key=lambda x: x[1], reverse=True)
            sector_results[sector] = similarities[:k * 3]

        # Collect unique memory IDs
        memory_ids = set()
        for results in sector_results.values():
            for mem_id, _ in results[:8]:
                memory_ids.add(mem_id)

        # Expand via waypoints if confidence is low
        avg_similarity = sum(
            sim for results in sector_results.values()
            for _, sim in results[:8]
        ) / max(1, sum(len(r[:8]) for r in sector_results.values()))

        if avg_similarity < 0.55:
            expansions = await self.graph.expand_via_waypoints(
                list(memory_ids),
                max_expansions=k * 2
            )
            for exp in expansions:
                memory_ids.add(exp.id)

        # Score and rank all candidates
        results = []

        for mem_id in memory_ids:
            # Get memory
            memory = await self.storage.get_memory(mem_id)
            if not memory:
                continue

            # Apply filters
            if user_id and memory['user_id'] != user_id:
                continue
            if min_salience and memory['salience'] < min_salience:
                continue

            # Calculate similarity score (best across sectors)
            best_similarity = 0.0
            best_sector = memory['primary_sector']

            for sector, results_list in sector_results.items():
                for mid, sim in results_list:
                    if mid == mem_id and sim > best_similarity:
                        best_similarity = sim
                        best_sector = sector.value

            # Calculate decay
            days_since = (int(time.time() * 1000) - memory['last_seen_at']) / (1000 * 60 * 60 * 24)
            decayed_salience = calc_decay(
                SectorType(memory['primary_sector']),
                memory['salience'],
                days_since
            )

            # Token overlap
            mem_tokens = canonical_token_set(memory['content'])
            token_overlap = compute_token_overlap(query_tokens, mem_tokens)

            # Recency score
            recency = calc_recency_score(memory['last_seen_at'])

            # Waypoint weight (from expansion if applicable)
            waypoint_weight = 0.0
            # TODO: Get from expansion results

            # Compute hybrid score
            final_score = compute_hybrid_score(
                best_similarity,
                token_overlap,
                waypoint_weight,
                recency
            )

            # Get sectors for this memory
            mem_vectors = await self.storage.get_vectors_by_id(mem_id)
            mem_sectors = [v['sector'] for v in mem_vectors]

            results.append(MemoryResult(
                id=mem_id,
                content=memory['content'],
                score=final_score,
                sectors=mem_sectors,
                primary_sector=memory['primary_sector'],
                path=[mem_id],
                salience=decayed_salience,
                last_seen_at=memory['last_seen_at']
            ))

        # Sort by score
        results.sort(key=lambda x: x.score, reverse=True)

        # Take top K
        top_results = results[:k]

        # Apply retrieval reinforcement
        now = int(time.time() * 1000)
        for result in top_results:
            new_salience = apply_retrieval_reinforcement(result.id, result.salience)
            await self.storage.update_memory_seen(
                result.id,
                now,
                new_salience,
                now
            )

        return top_results

    async def reinforce_memory(self, memory_id: str, boost: float = 0.1) -> None:
        """
        Manually reinforce a memory.

        Args:
            memory_id: Memory ID
            boost: Salience boost amount
        """
        memory = await self.storage.get_memory(memory_id)
        if not memory:
            raise ValueError(f"Memory {memory_id} not found")

        new_salience = min(1.0, memory['salience'] + boost)
        now = int(time.time() * 1000)

        await self.storage.update_memory_seen(
            memory_id,
            now,
            new_salience,
            now
        )

    def close(self) -> None:
        """Close system and cleanup"""
        self.storage.close()
