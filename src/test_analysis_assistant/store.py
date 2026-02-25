"""Persistent vector store for RAG corpus.

This module provides file-based persistence for retrieval chunks and embeddings,
enabling the corpus to persist between runs.
"""

import json
import math
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .retrieval import Chunk, SourceType


@dataclass
class ChunkMetadata:
    """Extended metadata for stored chunks."""
    chunk_id: str
    source_id: str
    source_type: str
    modality: str
    token_count: int
    created_at: str
    last_retrieved: Optional[str] = None
    retrieval_count: int = 0
    provenance: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StoredChunk:
    """A chunk stored with its metadata and optional embedding."""
    chunk_id: str
    source_id: str
    source_type: str
    modality: str
    text: str
    token_count: int
    metadata: Dict[str, Any]
    created_at: str
    embedding: Optional[List[float]] = None
    last_retrieved: Optional[str] = None
    retrieval_count: int = 0


@dataclass
class CorpusStats:
    """Statistics about the stored corpus."""
    total_chunks: int
    total_sources: int
    source_types: Dict[str, int]
    modalities: Dict[str, int]
    total_tokens: int
    embedding_dimension: Optional[int]
    created_at: str
    last_updated: str


class PersistentVectorStore:
    """File-based persistent vector store for RAG corpus.

    This store saves chunks and embeddings to JSON files, enabling
    the corpus to persist between runs. It also tracks retrieval
    statistics for confidence calibration.
    """

    DEFAULT_DIR = Path.home() / ".test_analysis_assistant" / "corpus"

    def __init__(
        self,
        storage_dir: Optional[str] = None,
        enable_embeddings: bool = True,
    ) -> None:
        """Initialize the persistent store.

        Args:
            storage_dir: Directory to store corpus files (default: ~/.test_analysis_assistant/corpus)
            enable_embeddings: Whether to store embeddings
        """
        self._storage_dir = Path(storage_dir) if storage_dir else self.DEFAULT_DIR
        self._enable_embeddings = enable_embeddings
        self._chunks: Dict[str, StoredChunk] = {}
        self._source_index: Dict[str, List[str]] = {}  # source_id -> chunk_ids
        self._embedding_dim: Optional[int] = None
        self._created_at = datetime.utcnow().isoformat()
        self._last_updated = self._created_at

    @property
    def storage_dir(self) -> Path:
        """Get the storage directory path."""
        return self._storage_dir

    def add_chunk(
        self,
        chunk: Chunk,
        embedding: Optional[List[float]] = None,
        provenance: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Add a chunk to the store.

        Args:
            chunk: The chunk to store
            embedding: Optional embedding vector
            provenance: Optional provenance information
        """
        stored = StoredChunk(
            chunk_id=chunk.chunk_id,
            source_id=chunk.source_id,
            source_type=chunk.source_type.value,
            modality=chunk.modality,
            text=chunk.text,
            token_count=chunk.token_count,
            metadata=chunk.metadata,
            embedding=embedding if self._enable_embeddings else None,
            created_at=datetime.utcnow().isoformat(),
            retrieval_count=0,
        )

        if provenance:
            stored.metadata["provenance"] = provenance

        self._chunks[chunk.chunk_id] = stored

        # Update source index
        if chunk.source_id not in self._source_index:
            self._source_index[chunk.source_id] = []
        if chunk.chunk_id not in self._source_index[chunk.source_id]:
            self._source_index[chunk.source_id].append(chunk.chunk_id)

        # Track embedding dimension
        if embedding:
            self._embedding_dim = len(embedding)

        self._last_updated = datetime.utcnow().isoformat()

    def add_chunks(
        self,
        chunks: Sequence[Chunk],
        embeddings: Optional[Dict[str, List[float]]] = None,
    ) -> int:
        """Add multiple chunks to the store.

        Args:
            chunks: Sequence of chunks to store
            embeddings: Optional dict mapping chunk_id to embedding vectors

        Returns:
            Number of chunks added
        """
        count = 0
        for chunk in chunks:
            embedding = embeddings.get(chunk.chunk_id) if embeddings else None
            self.add_chunk(chunk, embedding=embedding)
            count += 1
        return count

    def get_chunk(self, chunk_id: str) -> Optional[StoredChunk]:
        """Retrieve a chunk by ID.

        Args:
            chunk_id: The chunk ID

        Returns:
            The stored chunk, or None if not found
        """
        return self._chunks.get(chunk_id)

    def get_chunks_by_source(self, source_id: str) -> List[StoredChunk]:
        """Get all chunks from a specific source.

        Args:
            source_id: The source identifier

        Returns:
            List of stored chunks from that source
        """
        chunk_ids = self._source_index.get(source_id, [])
        return [self._chunks[cid] for cid in chunk_ids if cid in self._chunks]

    def query_by_text(
        self,
        query_text: str,
        top_k: int = 5,
    ) -> List[StoredChunk]:
        """Query chunks by text similarity (simple token overlap).

        Args:
            query_text: The query text
            top_k: Number of results to return

        Returns:
            List of matching chunks sorted by relevance
        """
        query_tokens = set(self._tokenize(query_text))
        if not query_tokens:
            return []

        scored: List[tuple] = []
        for chunk in self._chunks.values():
            chunk_tokens = set(self._tokenize(chunk.text))
            overlap = len(query_tokens.intersection(chunk_tokens))
            if overlap > 0:
                # Update retrieval stats
                chunk.retrieval_count += 1
                chunk.last_retrieved = datetime.utcnow().isoformat()
                scored.append((chunk, overlap / len(query_tokens)))

        scored.sort(key=lambda x: -x[1])
        return [chunk for chunk, _ in scored[:top_k]]

    def update_retrieval_stats(self, chunk_id: str) -> None:
        """Update retrieval statistics for a chunk.

        Args:
            chunk_id: The chunk ID that was retrieved
        """
        if chunk_id in self._chunks:
            self._chunks[chunk_id].retrieval_count += 1
            self._chunks[chunk_id].last_retrieved = datetime.utcnow().isoformat()

    def get_stats(self) -> CorpusStats:
        """Get statistics about the corpus.

        Returns:
            CorpusStats with aggregate information
        """
        source_types: Dict[str, int] = {}
        modalities: Dict[str, int] = {}
        total_tokens = 0

        for chunk in self._chunks.values():
            source_types[chunk.source_type] = source_types.get(chunk.source_type, 0) + 1
            modalities[chunk.modality] = modalities.get(chunk.modality, 0) + 1
            total_tokens += chunk.token_count

        return CorpusStats(
            total_chunks=len(self._chunks),
            total_sources=len(self._source_index),
            source_types=source_types,
            modalities=modalities,
            total_tokens=total_tokens,
            embedding_dimension=self._embedding_dim,
            created_at=self._created_at,
            last_updated=self._last_updated,
        )

    def save(self, filename: Optional[str] = None) -> str:
        """Save the corpus to a JSON file.

        Args:
            filename: Optional filename (default: corpus.json)

        Returns:
            Path to the saved file
        """
        self._storage_dir.mkdir(parents=True, exist_ok=True)
        filepath = self._storage_dir / (filename or "corpus.json")

        data = {
            "version": "1.0",
            "created_at": self._created_at,
            "last_updated": self._last_updated,
            "embedding_dim": self._embedding_dim,
            "chunks": [asdict(chunk) for chunk in self._chunks.values()],
            "source_index": self._source_index,
        }

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        return str(filepath)

    def load(self, filename: Optional[str] = None) -> int:
        """Load the corpus from a JSON file.

        Args:
            filename: Optional filename (default: corpus.json)

        Returns:
            Number of chunks loaded
        """
        filepath = self._storage_dir / (filename or "corpus.json")
        if not filepath.exists():
            return 0

        with open(filepath, encoding="utf-8") as f:
            data = json.load(f)

        self._created_at = data.get("created_at", datetime.utcnow().isoformat())
        self._last_updated = data.get("last_updated", datetime.utcnow().isoformat())
        self._embedding_dim = data.get("embedding_dim")

        self._chunks.clear()
        self._source_index.clear()

        for chunk_data in data.get("chunks", []):
            chunk = StoredChunk(**chunk_data)
            self._chunks[chunk.chunk_id] = chunk

            if chunk.source_id not in self._source_index:
                self._source_index[chunk.source_id] = []
            self._source_index[chunk.source_id].append(chunk.chunk_id)

        return len(self._chunks)

    def clear(self) -> None:
        """Clear all stored chunks."""
        self._chunks.clear()
        self._source_index.clear()
        self._embedding_dim = None
        self._last_updated = datetime.utcnow().isoformat()

    def get_chunks(self) -> List[StoredChunk]:
        """Get all stored chunks.

        Returns:
            List of all stored chunks
        """
        return list(self._chunks.values())

    def get_retrieval_stats(self) -> Dict[str, Any]:
        """Get retrieval statistics for all chunks.

        Returns:
            Dict with retrieval frequency info
        """
        stats = {
            "total_retrievals": sum(c.retrieval_count for c in self._chunks.values()),
            "chunks_retrieved": sum(1 for c in self._chunks.values() if c.retrieval_count > 0),
            "top_chunks": [],
        }

        sorted_chunks = sorted(
            self._chunks.values(),
            key=lambda c: c.retrieval_count,
            reverse=True
        )
        for chunk in sorted_chunks[:10]:
            stats["top_chunks"].append({
                "chunk_id": chunk.chunk_id,
                "source_id": chunk.source_id,
                "retrieval_count": chunk.retrieval_count,
            })

        return stats

    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization."""
        import re
        return re.findall(r"[a-zA-Z0-9_]+", text.lower())


class AdaptiveConfidenceCalibrator:
    """Calibrates confidence scores based on retrieval statistics.

    This calibrator uses retrieval frequency and source quality to
    provide more accurate confidence scores for retrieved chunks.
    """

    def __init__(self, store: PersistentVectorStore) -> None:
        self._store = store

    def calibrate(
        self,
        base_confidence: float,
        chunk_id: str,
    ) -> float:
        """Calibrate confidence based on retrieval history.

        Args:
            base_confidence: The base confidence from retrieval
            chunk_id: The chunk ID being calibrated

        Returns:
            Calibrated confidence score
        """
        chunk = self._store.get_chunk(chunk_id)
        if not chunk:
            return base_confidence

        # Factor 1: Retrieval frequency bonus (chunks retrieved often are more reliable)
        retrieval_factor = min(1.0, math.log2(chunk.retrieval_count + 2) / 5.0)

        # Factor 2: Provenance quality
        provenance = chunk.metadata.get("provenance", {})
        provenance_quality = provenance.get("quality_score", 0.7)

        # Factor 3: Source type reliability
        source_weights = {
            "requirements": 1.0,
            "system_analysis": 0.95,
            "code_snippet": 0.9,
            "repository": 0.85,
            "knowledge": 0.7,
        }
        source_factor = source_weights.get(chunk.source_type, 0.75)

        # Factor 4: Extraction confidence from original chunk
        extraction_confidence = chunk.metadata.get("extraction_confidence", 0.7)

        # Combine factors with calibrated weights
        calibrated = (
            base_confidence * 0.40
            + retrieval_factor * 0.15
            + provenance_quality * 0.15
            + source_factor * 0.15
            + extraction_confidence * 0.15
        )

        return round(max(0.0, min(1.0, calibrated)), 4)

    def get_reliability_score(self, chunk_id: str) -> float:
        """Get overall reliability score for a chunk.

        Args:
            chunk_id: The chunk ID

        Returns:
            Reliability score between 0 and 1
        """
        chunk = self._store.get_chunk(chunk_id)
        if not chunk:
            return 0.0

        # Retrieval count contributes to reliability
        retrieval_score = min(1.0, chunk.retrieval_count / 10.0)

        # Source reliability
        source_weights = {
            "requirements": 1.0,
            "system_analysis": 0.95,
            "code_snippet": 0.9,
            "repository": 0.85,
            "knowledge": 0.7,
        }
        source_score = source_weights.get(chunk.source_type, 0.75)

        # Metadata quality
        metadata_score = chunk.metadata.get("extraction_confidence", 0.7)

        return round((retrieval_score * 0.3 + source_score * 0.4 + metadata_score * 0.3), 4)


def create_persistent_engine(
    storage_dir: Optional[str] = None,
    chunk_size: int = 360,
    chunk_overlap: int = 40,
    enable_embeddings: bool = True,
):
    """Factory to create a retrieval engine with persistent storage.

    Args:
        storage_dir: Directory for corpus storage
        chunk_size: Maximum tokens per chunk
        chunk_overlap: Token overlap between chunks
        enable_embeddings: Whether to store embeddings

    Returns:
        Tuple of (HybridRetrievalEngine, PersistentVectorStore, AdaptiveConfidenceCalibrator)
    """
    from .retrieval import (
        DummyEmbeddingProvider,
        HybridRetrievalEngine,
        create_hybrid_engine,
    )

    engine = create_hybrid_engine(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    store = PersistentVectorStore(
        storage_dir=storage_dir,
        enable_embeddings=enable_embeddings,
    )
    calibrator = AdaptiveConfidenceCalibrator(store)

    return engine, store, calibrator
