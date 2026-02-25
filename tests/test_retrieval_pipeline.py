import unittest
import tempfile
from pathlib import Path

from src.test_analysis_assistant.retrieval import (
    IngestDocument,
    MultiSourceIngestor,
    QueryPlan,
    RetrievalEngine,
    SourceType,
    build_analysis_prompt,
)


class TestRetrievalPipeline(unittest.TestCase):
    def test_multisource_ingestor_reads_repository_files(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "src").mkdir()
            (root / "src" / "service.py").write_text(
                "def deploy_guard(failures):\n    return failures == 0\n",
                encoding="utf-8",
            )
            (root / "README.md").write_text(
                "Release is blocked when smoke tests fail.",
                encoding="utf-8",
            )
            (root / "image.png").write_bytes(b"\x89PNG\r\n\x1a\n")

            engine = RetrievalEngine()
            ingestor = MultiSourceIngestor(engine)
            chunks = ingestor.ingest_repository(str(root), max_files=10)

            self.assertGreaterEqual(len(chunks), 2)
            source_ids = {chunk.source_id for chunk in chunks}
            self.assertIn("repo:README.md", source_ids)
            self.assertIn("repo:src/service.py", source_ids)

    def test_markdown_requirements_extracts_table_and_image_units(self):
        markdown = """
# Auth Requirements
Missing negative authorization tests are release blocking.

| risk | severity |
| ---- | -------- |
| auth | high     |

![failure matrix](artifacts/failure-matrix.png)
""".strip()

        engine = RetrievalEngine()
        ingestor = MultiSourceIngestor(engine)
        chunks = ingestor.ingest_requirements_markdown("req-md-1", markdown)

        modalities = {chunk.modality for chunk in chunks}
        self.assertIn("text", modalities)
        self.assertIn("table", modalities)
        self.assertIn("image_ocr_stub", modalities)

    def test_ingest_multisource_with_table_and_image_stub(self):
        docs = [
            IngestDocument(
                source_id="req-1",
                source_type=SourceType.REQUIREMENTS,
                content="System shall block deployment when smoke tests fail.",
            ),
            IngestDocument(
                source_id="table-1",
                source_type=SourceType.SYSTEM_ANALYSIS,
                content={"rows": [{"risk": "auth", "severity": "high"}]},
                modality="table",
            ),
            IngestDocument(
                source_id="img-1",
                source_type=SourceType.SYSTEM_ANALYSIS,
                content={"image_path": "diagram.png"},
                modality="image",
            ),
        ]

        engine = RetrievalEngine()
        chunks = engine.ingest_documents(docs)

        self.assertGreaterEqual(len(chunks), 3)
        modalities = {chunk.modality for chunk in chunks}
        self.assertIn("text", modalities)
        self.assertIn("table", modalities)
        self.assertIn("image_ocr_stub", modalities)

    def test_query_ranks_goal_aligned_requirement_first(self):
        docs = [
            IngestDocument(
                source_id="req-auth",
                source_type=SourceType.REQUIREMENTS,
                content="Authentication regressions are release blocking and require P0 triage.",
            ),
            IngestDocument(
                source_id="note-gfx",
                source_type=SourceType.KNOWLEDGE,
                content="UI color palette update and typography polish for dashboard cards.",
            ),
        ]

        engine = RetrievalEngine()
        engine.ingest_documents(docs)
        ranked = engine.query("release blocking authentication bug triage", top_k=2)

        self.assertEqual(len(ranked), 2)
        self.assertEqual(ranked[0].chunk.source_id, "req-auth")
        self.assertGreater(ranked[0].score, ranked[1].score)

    def test_confidence_is_bounded_and_descends_with_rank(self):
        docs = [
            IngestDocument(
                source_id="req-1",
                source_type=SourceType.REQUIREMENTS,
                content="Root-cause analysis should prioritize flaky integration tests.",
            ),
            IngestDocument(
                source_id="doc-2",
                source_type=SourceType.SYSTEM_ANALYSIS,
                content="Performance benchmarks for rendering path.",
            ),
        ]

        engine = RetrievalEngine()
        engine.ingest_documents(docs)
        ranked = engine.query("prioritize flaky tests", top_k=2)

        self.assertTrue(0.0 <= ranked[0].confidence <= 1.0)
        self.assertTrue(0.0 <= ranked[1].confidence <= 1.0)
        self.assertGreaterEqual(ranked[0].confidence, ranked[1].confidence)

    def test_prompt_includes_citations_and_confidence(self):
        docs = [
            IngestDocument(
                source_id="req-1",
                source_type=SourceType.REQUIREMENTS,
                content="Test-gap analysis must include missing negative authorization tests.",
            )
        ]
        engine = RetrievalEngine()
        engine.ingest_documents(docs)
        ranked = engine.query("negative authorization tests", top_k=1)

        prompt = build_analysis_prompt(
            question="What are highest-risk gaps?",
            ranked_context=ranked,
        )

        self.assertIn("What are highest-risk gaps?", prompt)
        self.assertIn("confidence=", prompt)
        self.assertIn("req-1", prompt)

    def test_build_query_plan_detects_risk_gap_intent(self):
        engine = RetrievalEngine()
        plan = engine.build_query_plan("Identify test gaps and prioritize release risks.")

        self.assertIsInstance(plan, QueryPlan)
        self.assertIn("test_gap", plan.intent_labels)
        self.assertIn("risk_prioritization", plan.intent_labels)
        self.assertIn(SourceType.REQUIREMENTS, plan.preferred_source_types)

    def test_query_exposes_score_breakdown(self):
        docs = [
            IngestDocument(
                source_id="req-1",
                source_type=SourceType.REQUIREMENTS,
                content="Missing negative authorization tests create release risk.",
            ),
            IngestDocument(
                source_id="img-1",
                source_type=SourceType.SYSTEM_ANALYSIS,
                content={"image_path": "risk.png"},
                modality="image",
            ),
        ]

        engine = RetrievalEngine()
        engine.ingest_documents(docs)
        ranked = engine.query("negative authorization test gap risk", top_k=2)

        self.assertEqual(ranked[0].chunk.source_id, "req-1")
        self.assertIn("lexical", ranked[0].score_breakdown)
        self.assertIn("source", ranked[0].score_breakdown)
        self.assertIn("intent", ranked[0].score_breakdown)
        self.assertIn("extraction", ranked[0].score_breakdown)
        self.assertGreater(ranked[0].score_breakdown["extraction"], ranked[1].score_breakdown["extraction"])

    def test_diversify_prefers_cross_source_coverage(self):
        docs = [
            IngestDocument(
                source_id="repo-a",
                source_type=SourceType.REPOSITORY,
                content=(
                    "authentication failure test gap release risk blocking "
                    "authorization negative path triage"
                ),
            ),
            IngestDocument(
                source_id="repo-a",
                source_type=SourceType.REPOSITORY,
                content=(
                    "authentication failure test gap release risk blocking "
                    "authorization token expiry scenario"
                ),
            ),
            IngestDocument(
                source_id="req-b",
                source_type=SourceType.REQUIREMENTS,
                content="release risk requires P0 triage for auth test gap",
            ),
        ]
        engine = RetrievalEngine()
        engine.ingest_documents(docs)

        diverse = engine.query("auth test gap release risk triage", top_k=2, diversify=True)
        plain = engine.query("auth test gap release risk triage", top_k=2, diversify=False)

        diverse_sources = {item.chunk.source_id for item in diverse}
        plain_sources = {item.chunk.source_id for item in plain}
        self.assertGreaterEqual(len(diverse_sources), 2)
        self.assertLessEqual(len(plain_sources), 2)


if __name__ == "__main__":
    unittest.main()
