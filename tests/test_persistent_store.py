"""Tests for persistent vector store."""

import os
import tempfile
import unittest
from pathlib import Path

from src.test_analysis_assistant.retrieval import Chunk, SourceType
from src.test_analysis_assistant.store import (
    AdaptiveConfidenceCalibrator,
    CorpusStats,
    PersistentVectorStore,
    StoredChunk,
)


class TestPersistentVectorStore(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.store = PersistentVectorStore(storage_dir=self.temp_dir)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_add_chunk(self):
        chunk = Chunk(
            chunk_id="test_1",
            source_id="repo:main.py",
            source_type=SourceType.REPOSITORY,
            modality="code",
            text="def hello(): return 'world'",
            token_count=5,
            metadata={"language": "python"},
        )
        self.store.add_chunk(chunk)

        retrieved = self.store.get_chunk("test_1")
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.text, "def hello(): return 'world'")

    def test_add_chunks(self):
        chunks = [
            Chunk(
                chunk_id=f"chunk_{i}",
                source_id="repo:main.py",
                source_type=SourceType.REPOSITORY,
                modality="code",
                text=f"code line {i}",
                token_count=2,
                metadata={},
            )
            for i in range(5)
        ]
        count = self.store.add_chunks(chunks)
        self.assertEqual(count, 5)
        self.assertEqual(len(self.store.get_chunks()), 5)

    def test_get_chunks_by_source(self):
        chunks = [
            Chunk(
                chunk_id="chunk_1",
                source_id="repo:module.py",
                source_type=SourceType.REPOSITORY,
                modality="code",
                text="code 1",
                token_count=1,
                metadata={},
            ),
            Chunk(
                chunk_id="chunk_2",
                source_id="repo:other.py",
                source_type=SourceType.REPOSITORY,
                modality="code",
                text="code 2",
                token_count=1,
                metadata={},
            ),
        ]
        self.store.add_chunks(chunks)

        module_chunks = self.store.get_chunks_by_source("repo:module.py")
        self.assertEqual(len(module_chunks), 1)
        self.assertEqual(module_chunks[0].chunk_id, "chunk_1")

    def test_query_by_text(self):
        chunks = [
            Chunk(
                chunk_id="chunk_1",
                source_id="repo:auth.py",
                source_type=SourceType.REPOSITORY,
                modality="code",
                text="def authenticate(user, password): return login",
                token_count=6,
                metadata={},
            ),
            Chunk(
                chunk_id="chunk_2",
                source_id="repo:utils.py",
                source_type=SourceType.REPOSITORY,
                modality="code",
                text="def parse_json(data): return json_loads",
                token_count=6,
                metadata={},
            ),
        ]
        self.store.add_chunks(chunks)

        # Use exact token match
        results = self.store.query_by_text("authenticate login", top_k=1)
        self.assertGreaterEqual(len(results), 1)

    def test_save_and_load(self):
        chunks = [
            Chunk(
                chunk_id="chunk_1",
                source_id="repo:main.py",
                source_type=SourceType.REPOSITORY,
                modality="code",
                text="def main(): pass",
                token_count=3,
                metadata={"language": "python"},
            ),
        ]
        embeddings = {"chunk_1": [0.1, 0.2, 0.3]}
        self.store.add_chunks(chunks, embeddings=embeddings)

        filepath = self.store.save("test_corpus.json")

        new_store = PersistentVectorStore(storage_dir=self.temp_dir)
        loaded_count = new_store.load("test_corpus.json")

        self.assertEqual(loaded_count, 1)
        loaded_chunk = new_store.get_chunk("chunk_1")
        self.assertIsNotNone(loaded_chunk)
        self.assertEqual(loaded_chunk.text, "def main(): pass")

    def test_get_stats(self):
        chunks = [
            Chunk(
                chunk_id=f"chunk_{i}",
                source_id=f"repo:file{i}.py",
                source_type=SourceType.REPOSITORY,
                modality="code" if i < 3 else "text",
                text=f"code {i}" * 10,
                token_count=10,
                metadata={},
            )
            for i in range(5)
        ]
        self.store.add_chunks(chunks)

        stats = self.store.get_stats()
        self.assertIsInstance(stats, CorpusStats)
        self.assertEqual(stats.total_chunks, 5)
        self.assertEqual(stats.total_sources, 5)

    def test_retrieval_stats_update(self):
        chunk = Chunk(
            chunk_id="chunk_1",
            source_id="repo:main.py",
            source_type=SourceType.REPOSITORY,
            modality="code",
            text="code",
            token_count=1,
            metadata={},
        )
        self.store.add_chunk(chunk)

        self.store.update_retrieval_stats("chunk_1")
        self.store.update_retrieval_stats("chunk_1")

        retrieved = self.store.get_chunk("chunk_1")
        self.assertEqual(retrieved.retrieval_count, 2)

    def test_provenance_tracking(self):
        chunk = Chunk(
            chunk_id="chunk_1",
            source_id="repo:main.py",
            source_type=SourceType.REPOSITORY,
            modality="code",
            text="def test(): pass",
            token_count=3,
            metadata={},
        )
        provenance = {
            "file_path": "/path/to/main.py",
            "git_commit": "abc123",
            "quality_score": 0.95,
        }
        self.store.add_chunk(chunk, provenance=provenance)

        stored = self.store.get_chunk("chunk_1")
        self.assertEqual(stored.metadata.get("provenance", {}).get("quality_score"), 0.95)

    def test_clear(self):
        chunk = Chunk(
            chunk_id="chunk_1",
            source_id="repo:main.py",
            source_type=SourceType.REPOSITORY,
            modality="code",
            text="code",
            token_count=1,
            metadata={},
        )
        self.store.add_chunk(chunk)
        self.assertEqual(len(self.store.get_chunks()), 1)

        self.store.clear()
        self.assertEqual(len(self.store.get_chunks()), 0)


class TestAdaptiveConfidenceCalibrator(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.store = PersistentVectorStore(storage_dir=self.temp_dir)
        self.calibrator = AdaptiveConfidenceCalibrator(self.store)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_calibrate_with_retrieval_history(self):
        chunk = Chunk(
            chunk_id="chunk_1",
            source_id="requirements:SPEC.md",
            source_type=SourceType.REQUIREMENTS,
            modality="text",
            text="The system shall authenticate users",
            token_count=6,
            metadata={"extraction_confidence": 0.95},
        )
        self.store.add_chunk(chunk)

        # Simulate multiple retrievals
        self.store.update_retrieval_stats("chunk_1")
        self.store.update_retrieval_stats("chunk_1")
        self.store.update_retrieval_stats("chunk_1")

        calibrated = self.calibrator.calibrate(0.7, "chunk_1")
        self.assertGreater(calibrated, 0.7)  # Should be boosted by retrieval history

    def test_calibrate_missing_chunk(self):
        calibrated = self.calibrator.calibrate(0.5, "nonexistent")
        self.assertEqual(calibrated, 0.5)  # Falls back to base confidence

    def test_reliability_score(self):
        chunk = Chunk(
            chunk_id="chunk_1",
            source_id="requirements:SPEC.md",
            source_type=SourceType.REQUIREMENTS,
            modality="text",
            text="The system shall authenticate users",
            token_count=6,
            metadata={"extraction_confidence": 0.95},
        )
        self.store.add_chunk(chunk)

        # Simulate multiple retrievals to boost reliability
        for _ in range(5):
            self.store.update_retrieval_stats("chunk_1")

        reliability = self.calibrator.get_reliability_score("chunk_1")
        self.assertGreater(reliability, 0.75)  # Should be boosted by retrieval history + high extraction


if __name__ == "__main__":
    unittest.main()
