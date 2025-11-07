"""
Waypoint graph for memory associations
"""
import time
from typing import List, Dict, Set, Optional, Tuple
from dataclasses import dataclass


@dataclass
class WaypointExpansion:
    """Result of waypoint expansion"""
    id: str
    weight: float
    path: List[str]


class WaypointGraph:
    """
    Manages waypoint associations between memories.

    Waypoints are weighted, bidirectional links that represent
    associative connections between memories.
    """

    def __init__(self, storage):
        """
        Initialize waypoint graph.

        Args:
            storage: Storage backend for persistence
        """
        self.storage = storage

    async def create_waypoint(
        self,
        src_id: str,
        dst_id: str,
        weight: float,
        bidirectional: bool = True
    ) -> None:
        """
        Create a waypoint between two memories.

        Args:
            src_id: Source memory ID
            dst_id: Destination memory ID
            weight: Association weight (0-1)
            bidirectional: If True, creates both directions
        """
        now = int(time.time() * 1000)

        # Create forward waypoint
        await self.storage.insert_waypoint(src_id, dst_id, weight, now, now)

        # Create reverse waypoint if bidirectional
        if bidirectional:
            await self.storage.insert_waypoint(dst_id, src_id, weight, now, now)

    async def update_waypoint_weight(
        self,
        src_id: str,
        dst_id: str,
        new_weight: float
    ) -> None:
        """
        Update waypoint weight.

        Args:
            src_id: Source memory ID
            dst_id: Destination memory ID
            new_weight: New weight value
        """
        now = int(time.time() * 1000)
        await self.storage.update_waypoint(src_id, dst_id, new_weight, now)

    async def get_neighbors(self, memory_id: str) -> List[Tuple[str, float]]:
        """
        Get all neighbors of a memory.

        Args:
            memory_id: Memory ID

        Returns:
            List of (neighbor_id, weight) tuples
        """
        waypoints = await self.storage.get_waypoints_by_source(memory_id)
        return [(wp['dst_id'], wp['weight']) for wp in waypoints]

    async def expand_via_waypoints(
        self,
        initial_results: List[str],
        max_expansions: int = 10
    ) -> List[WaypointExpansion]:
        """
        Expand search results via waypoint traversal.

        Uses breadth-first search to find associated memories.

        Args:
            initial_results: List of initial memory IDs
            max_expansions: Maximum number of expansions

        Returns:
            List of expansions with weights and paths
        """
        expansions: List[WaypointExpansion] = []
        visited: Set[str] = set()
        queue: List[WaypointExpansion] = []

        # Initialize with starting nodes
        for mem_id in initial_results:
            expansion = WaypointExpansion(
                id=mem_id,
                weight=1.0,
                path=[mem_id]
            )
            expansions.append(expansion)
            visited.add(mem_id)
            queue.append(expansion)

        expansion_count = 0

        # BFS expansion
        while queue and expansion_count < max_expansions:
            current = queue.pop(0)

            # Get neighbors
            neighbors = await self.get_neighbors(current.id)

            for neighbor_id, neighbor_weight in neighbors:
                if neighbor_id in visited:
                    continue

                # Calculate expanded weight (decay by distance)
                expanded_weight = current.weight * neighbor_weight * 0.8

                # Skip weak associations
                if expanded_weight < 0.1:
                    continue

                # Create expansion
                expansion = WaypointExpansion(
                    id=neighbor_id,
                    weight=expanded_weight,
                    path=current.path + [neighbor_id]
                )

                expansions.append(expansion)
                visited.add(neighbor_id)
                queue.append(expansion)
                expansion_count += 1

                if expansion_count >= max_expansions:
                    break

        return expansions

    async def reinforce_path(
        self,
        path: List[str],
        boost: float = 0.05
    ) -> None:
        """
        Reinforce waypoints along a traversed path.

        Args:
            path: List of memory IDs in traversal order
            boost: Weight boost to apply
        """
        now = int(time.time() * 1000)

        for i in range(len(path) - 1):
            src_id = path[i]
            dst_id = path[i + 1]

            # Get current waypoint
            waypoint = await self.storage.get_waypoint(src_id, dst_id)

            if waypoint:
                # Boost weight (max 1.0)
                new_weight = min(1.0, waypoint['weight'] + boost)
                await self.storage.update_waypoint(src_id, dst_id, new_weight, now)

    async def create_contextual_waypoints(
        self,
        memory_id: str,
        related_ids: List[str],
        base_weight: float = 0.3
    ) -> None:
        """
        Create contextual waypoints from a memory to related memories.

        Args:
            memory_id: Source memory ID
            related_ids: List of related memory IDs
            base_weight: Base weight for new waypoints
        """
        now = int(time.time() * 1000)

        for related_id in related_ids:
            if memory_id == related_id:
                continue

            # Check if waypoint exists
            existing = await self.storage.get_waypoint(memory_id, related_id)

            if existing:
                # Boost existing waypoint
                new_weight = min(1.0, existing['weight'] + 0.1)
                await self.storage.update_waypoint(memory_id, related_id, new_weight, now)
            else:
                # Create new waypoint
                await self.storage.insert_waypoint(memory_id, related_id, base_weight, now, now)

    async def prune_weak_waypoints(self, threshold: float = 0.05) -> int:
        """
        Remove weak waypoints below threshold.

        Args:
            threshold: Minimum weight to keep

        Returns:
            Number of waypoints pruned
        """
        return await self.storage.prune_waypoints(threshold)

    async def create_similarity_waypoint(
        self,
        new_id: str,
        new_vector,
        threshold: float = 0.75
    ) -> None:
        """
        Create waypoint to most similar existing memory.

        Args:
            new_id: New memory ID
            new_vector: New memory embedding vector
            threshold: Minimum similarity threshold
        """
        from .embeddings import cosine_similarity

        # Get all memories with mean vectors
        memories = await self.storage.get_all_memories_with_vectors(limit=1000)

        best_match = None
        best_similarity = 0.0

        for mem in memories:
            if mem['id'] == new_id or not mem['mean_vec']:
                continue

            # Calculate similarity
            similarity = cosine_similarity(new_vector, mem['mean_vec'])

            if similarity > best_similarity:
                best_similarity = similarity
                best_match = mem['id']

        # Create waypoint if above threshold
        if best_match and best_similarity >= threshold:
            now = int(time.time() * 1000)
            await self.storage.insert_waypoint(new_id, best_match, best_similarity, now, now)
        elif not best_match:
            # Self-referential waypoint if no match
            now = int(time.time() * 1000)
            await self.storage.insert_waypoint(new_id, new_id, 1.0, now, now)
