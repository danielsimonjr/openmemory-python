"""
SQLite storage backend for OpenMemory
"""
import sqlite3
import json
import asyncio
from typing import List, Dict, Any, Optional
from pathlib import Path
import numpy as np


class Storage:
    """
    SQLite storage backend for memories, vectors, and waypoints.
    """

    def __init__(self, db_path: str = "openmemory.db"):
        """
        Initialize storage.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self.conn: Optional[sqlite3.Connection] = None
        self._initialize_db()

    def _initialize_db(self) -> None:
        """Initialize database schema"""
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row

        # Create tables
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS memories (
                id TEXT PRIMARY KEY,
                user_id TEXT,
                segment INTEGER DEFAULT 0,
                content TEXT NOT NULL,
                simhash TEXT,
                primary_sector TEXT NOT NULL,
                tags TEXT,
                meta TEXT,
                created_at INTEGER NOT NULL,
                updated_at INTEGER NOT NULL,
                last_seen_at INTEGER NOT NULL,
                salience REAL DEFAULT 0.5,
                decay_lambda REAL DEFAULT 0.01,
                version INTEGER DEFAULT 1,
                mean_vec BLOB,
                mean_vec_dim INTEGER,
                compressed_vec BLOB,
                feedback_score REAL DEFAULT 0.0
            );

            CREATE INDEX IF NOT EXISTS idx_memories_user ON memories(user_id);
            CREATE INDEX IF NOT EXISTS idx_memories_sector ON memories(primary_sector);
            CREATE INDEX IF NOT EXISTS idx_memories_segment ON memories(segment);
            CREATE INDEX IF NOT EXISTS idx_memories_simhash ON memories(simhash);
            CREATE INDEX IF NOT EXISTS idx_memories_salience ON memories(salience);

            CREATE TABLE IF NOT EXISTS vectors (
                id TEXT NOT NULL,
                sector TEXT NOT NULL,
                vec BLOB NOT NULL,
                dim INTEGER NOT NULL,
                PRIMARY KEY (id, sector),
                FOREIGN KEY (id) REFERENCES memories(id) ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_vectors_sector ON vectors(sector);

            CREATE TABLE IF NOT EXISTS waypoints (
                src_id TEXT NOT NULL,
                dst_id TEXT NOT NULL,
                weight REAL NOT NULL,
                created_at INTEGER NOT NULL,
                updated_at INTEGER NOT NULL,
                PRIMARY KEY (src_id, dst_id),
                FOREIGN KEY (src_id) REFERENCES memories(id) ON DELETE CASCADE,
                FOREIGN KEY (dst_id) REFERENCES memories(id) ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_waypoints_src ON waypoints(src_id);
            CREATE INDEX IF NOT EXISTS idx_waypoints_dst ON waypoints(dst_id);
            CREATE INDEX IF NOT EXISTS idx_waypoints_weight ON waypoints(weight);
        """)

        self.conn.commit()

    async def insert_memory(
        self,
        id: str,
        content: str,
        primary_sector: str,
        salience: float,
        decay_lambda: float,
        created_at: int,
        user_id: Optional[str] = None,
        tags: Optional[str] = None,
        meta: Optional[str] = None,
        simhash: Optional[str] = None,
        segment: int = 0
    ) -> None:
        """Insert a new memory"""
        await asyncio.to_thread(
            self.conn.execute,
            """
            INSERT INTO memories (
                id, user_id, segment, content, simhash, primary_sector,
                tags, meta, created_at, updated_at, last_seen_at,
                salience, decay_lambda, version
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                id, user_id, segment, content, simhash, primary_sector,
                tags, meta, created_at, created_at, created_at,
                salience, decay_lambda, 1
            )
        )
        await asyncio.to_thread(self.conn.commit)

    async def get_memory(self, memory_id: str) -> Optional[Dict]:
        """Get a memory by ID"""
        cursor = await asyncio.to_thread(
            self.conn.execute,
            "SELECT * FROM memories WHERE id = ?",
            (memory_id,)
        )
        row = await asyncio.to_thread(cursor.fetchone)
        return dict(row) if row else None

    async def get_memory_by_simhash(self, simhash: str) -> Optional[Dict]:
        """Get a memory by simhash"""
        cursor = await asyncio.to_thread(
            self.conn.execute,
            "SELECT * FROM memories WHERE simhash = ?",
            (simhash,)
        )
        row = await asyncio.to_thread(cursor.fetchone)
        return dict(row) if row else None

    async def update_memory_seen(
        self,
        memory_id: str,
        last_seen_at: int,
        salience: float,
        updated_at: int
    ) -> None:
        """Update memory last seen and salience"""
        await asyncio.to_thread(
            self.conn.execute,
            """
            UPDATE memories
            SET last_seen_at = ?, salience = ?, updated_at = ?
            WHERE id = ?
            """,
            (last_seen_at, salience, updated_at, memory_id)
        )
        await asyncio.to_thread(self.conn.commit)

    async def insert_vector(
        self,
        memory_id: str,
        sector: str,
        vector: np.ndarray
    ) -> None:
        """Insert embedding vector"""
        vec_bytes = vector.astype(np.float32).tobytes()
        dim = len(vector)

        await asyncio.to_thread(
            self.conn.execute,
            "INSERT INTO vectors (id, sector, vec, dim) VALUES (?, ?, ?, ?)",
            (memory_id, sector, vec_bytes, dim)
        )
        await asyncio.to_thread(self.conn.commit)

    async def get_vectors_by_sector(self, sector: str) -> List[Dict]:
        """Get all vectors for a sector"""
        cursor = await asyncio.to_thread(
            self.conn.execute,
            "SELECT id, vec, dim FROM vectors WHERE sector = ?",
            (sector,)
        )
        rows = await asyncio.to_thread(cursor.fetchall)

        results = []
        for row in rows:
            vec_bytes = row['vec']
            vec = np.frombuffer(vec_bytes, dtype=np.float32)
            results.append({
                'id': row['id'],
                'v': vec,
                'dim': row['dim']
            })

        return results

    async def get_vectors_by_id(self, memory_id: str) -> List[Dict]:
        """Get all vectors for a memory"""
        cursor = await asyncio.to_thread(
            self.conn.execute,
            "SELECT sector, vec, dim FROM vectors WHERE id = ?",
            (memory_id,)
        )
        rows = await asyncio.to_thread(cursor.fetchall)

        results = []
        for row in rows:
            vec_bytes = row['vec']
            vec = np.frombuffer(vec_bytes, dtype=np.float32)
            results.append({
                'sector': row['sector'],
                'v': vec,
                'dim': row['dim']
            })

        return results

    async def update_mean_vector(
        self,
        memory_id: str,
        mean_vec: np.ndarray
    ) -> None:
        """Update mean vector for a memory"""
        vec_bytes = mean_vec.astype(np.float32).tobytes()
        dim = len(mean_vec)

        await asyncio.to_thread(
            self.conn.execute,
            "UPDATE memories SET mean_vec = ?, mean_vec_dim = ? WHERE id = ?",
            (vec_bytes, dim, memory_id)
        )
        await asyncio.to_thread(self.conn.commit)

    async def insert_waypoint(
        self,
        src_id: str,
        dst_id: str,
        weight: float,
        created_at: int,
        updated_at: int
    ) -> None:
        """Insert or replace waypoint"""
        await asyncio.to_thread(
            self.conn.execute,
            """
            INSERT OR REPLACE INTO waypoints (src_id, dst_id, weight, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (src_id, dst_id, weight, created_at, updated_at)
        )
        await asyncio.to_thread(self.conn.commit)

    async def get_waypoint(self, src_id: str, dst_id: str) -> Optional[Dict]:
        """Get a specific waypoint"""
        cursor = await asyncio.to_thread(
            self.conn.execute,
            "SELECT * FROM waypoints WHERE src_id = ? AND dst_id = ?",
            (src_id, dst_id)
        )
        row = await asyncio.to_thread(cursor.fetchone)
        return dict(row) if row else None

    async def get_waypoints_by_source(self, src_id: str) -> List[Dict]:
        """Get all waypoints from a source"""
        cursor = await asyncio.to_thread(
            self.conn.execute,
            "SELECT * FROM waypoints WHERE src_id = ?",
            (src_id,)
        )
        rows = await asyncio.to_thread(cursor.fetchall)
        return [dict(row) for row in rows]

    async def update_waypoint(
        self,
        src_id: str,
        dst_id: str,
        weight: float,
        updated_at: int
    ) -> None:
        """Update waypoint weight"""
        await asyncio.to_thread(
            self.conn.execute,
            """
            UPDATE waypoints
            SET weight = ?, updated_at = ?
            WHERE src_id = ? AND dst_id = ?
            """,
            (weight, updated_at, src_id, dst_id)
        )
        await asyncio.to_thread(self.conn.commit)

    async def prune_waypoints(self, threshold: float) -> int:
        """Remove waypoints below threshold"""
        cursor = await asyncio.to_thread(
            self.conn.execute,
            "DELETE FROM waypoints WHERE weight < ?",
            (threshold,)
        )
        deleted = cursor.rowcount
        await asyncio.to_thread(self.conn.commit)
        return deleted

    async def get_all_memories_with_vectors(self, limit: int = 1000) -> List[Dict]:
        """Get all memories with mean vectors"""
        cursor = await asyncio.to_thread(
            self.conn.execute,
            """
            SELECT id, mean_vec, mean_vec_dim
            FROM memories
            WHERE mean_vec IS NOT NULL
            LIMIT ?
            """,
            (limit,)
        )
        rows = await asyncio.to_thread(cursor.fetchall)

        results = []
        for row in rows:
            if row['mean_vec']:
                vec = np.frombuffer(row['mean_vec'], dtype=np.float32)
                results.append({
                    'id': row['id'],
                    'mean_vec': vec,
                    'dim': row['mean_vec_dim']
                })

        return results

    async def get_all_memories(self, limit: int, offset: int = 0) -> List[Dict]:
        """Get all memories with pagination"""
        cursor = await asyncio.to_thread(
            self.conn.execute,
            """
            SELECT * FROM memories
            ORDER BY last_seen_at DESC
            LIMIT ? OFFSET ?
            """,
            (limit, offset)
        )
        rows = await asyncio.to_thread(cursor.fetchall)
        return [dict(row) for row in rows]

    async def get_max_segment(self) -> int:
        """Get maximum segment number"""
        cursor = await asyncio.to_thread(
            self.conn.execute,
            "SELECT MAX(segment) as max_seg FROM memories"
        )
        row = await asyncio.to_thread(cursor.fetchone)
        return row['max_seg'] if row and row['max_seg'] is not None else 0

    async def get_segment_count(self, segment: int) -> int:
        """Get count of memories in segment"""
        cursor = await asyncio.to_thread(
            self.conn.execute,
            "SELECT COUNT(*) as c FROM memories WHERE segment = ?",
            (segment,)
        )
        row = await asyncio.to_thread(cursor.fetchone)
        return row['c'] if row else 0

    def close(self) -> None:
        """Close database connection"""
        if self.conn:
            self.conn.close()
