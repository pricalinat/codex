import unittest
import tempfile
from pathlib import Path

from src.test_analysis_assistant.retrieval import (
    DummyEmbeddingProvider,
    HybridRetrievalEngine,
    IngestDocument,
    MultiSourceIngestor,
    QueryPlan,
    RetrievalEngine,
    SourceType,
    TFIDFEmbeddingProvider,
    build_analysis_prompt,
    create_hybrid_engine,
)


class TestRetrievalPipeline(unittest.TestCase):
    def test_retrieve_evidence_reports_missing_modalities(self):
        docs = [
            IngestDocument(
                source_id="req-auth",
                source_type=SourceType.REQUIREMENTS,
                content="Authentication regressions are release blocking and require triage.",
            ),
        ]

        engine = RetrievalEngine()
        engine.ingest_documents(docs)

        evidence = engine.retrieve_evidence(
            "image table risk matrix for authentication gaps",
            top_k=2,
        )

        self.assertTrue(len(evidence.ranked_chunks) > 0)
        self.assertIn("table", evidence.missing_modalities)
        self.assertIn("image", evidence.missing_modalities)
        self.assertIn("image_ocr_stub", evidence.missing_modalities)
        self.assertIn(evidence.confidence_band, {"low", "medium", "high"})

    def test_query_with_expansion_brings_goal_complementary_source(self):
        docs = [
            IngestDocument(
                source_id="repo-traceback",
                source_type=SourceType.REPOSITORY,
                content="Traceback analysis shows root cause in authentication token parser.",
            ),
            IngestDocument(
                source_id="sys-debug",
                source_type=SourceType.SYSTEM_ANALYSIS,
                content="Failure diagnostics and traceback processing for flaky integration failures.",
            ),
            IngestDocument(
                source_id="req-plan",
                source_type=SourceType.REQUIREMENTS,
                content="Actionable mitigation plan and risk prioritization steps for release readiness.",
            ),
        ]

        engine = RetrievalEngine()
        engine.ingest_documents(docs)

        baseline = engine.query("traceback root cause for failing tests", top_k=2, diversify=False)
        expanded = engine.query_with_expansion(
            "traceback root cause for failing tests",
            top_k=3,
            diversify=False,
        )

        baseline_sources = {item.chunk.source_id for item in baseline}
        expanded_sources = {item.chunk.source_id for item in expanded}
        self.assertNotIn("req-plan", baseline_sources)
        self.assertIn("req-plan", expanded_sources)

    def test_retrieve_evidence_computes_aggregate_confidence(self):
        docs = [
            IngestDocument(
                source_id="req-auth",
                source_type=SourceType.REQUIREMENTS,
                content="Authentication risk requires prioritization and missing test coverage review.",
            ),
            IngestDocument(
                source_id="repo-auth",
                source_type=SourceType.REPOSITORY,
                content="Root cause in login flow with missing negative test paths.",
            ),
        ]
        engine = RetrievalEngine()
        engine.ingest_documents(docs)

        evidence = engine.retrieve_evidence("auth risk prioritization", top_k=2)

        self.assertGreaterEqual(evidence.aggregate_confidence, 0.0)
        self.assertLessEqual(evidence.aggregate_confidence, 1.0)
        self.assertIn(evidence.confidence_band, {"low", "medium", "high"})

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

    def test_compound_modality_extracts_multimodal_units_with_provenance(self):
        docs = [
            IngestDocument(
                source_id="compound-1",
                source_type=SourceType.SYSTEM_ANALYSIS,
                modality="compound",
                content={
                    "text": "Auth flow fails under high latency.",
                    "tables": [{"rows": [{"risk": "auth", "severity": "high"}]}],
                    "images": [
                        {
                            "image_path": "diag/auth-failure.png",
                            "ocr_text": "retry budget exceeded in auth gateway",
                        }
                    ],
                },
                metadata={"origin_path": "docs/system/auth.md"},
            ),
        ]

        engine = RetrievalEngine()
        chunks = engine.ingest_documents(docs)

        self.assertGreaterEqual(len(chunks), 3)
        modality_set = {chunk.modality for chunk in chunks}
        self.assertIn("text", modality_set)
        self.assertIn("table", modality_set)
        self.assertIn("image", modality_set)

        kinds = {chunk.metadata.get("unit_kind") for chunk in chunks}
        self.assertIn("text", kinds)
        self.assertIn("table", kinds)
        self.assertIn("image", kinds)
        self.assertTrue(all(chunk.metadata.get("origin_path") == "docs/system/auth.md" for chunk in chunks))

    def test_compound_ingestion_surfaces_image_and_table_coverage_in_evidence(self):
        docs = [
            IngestDocument(
                source_id="compound-coverage",
                source_type=SourceType.SYSTEM_ANALYSIS,
                modality="compound",
                content={
                    "text": "Failure clustering indicates auth retry storms.",
                    "tables": [{"rows": [{"component": "auth", "risk": "high"}]}],
                    "images": [{"ocr_text": "heatmap shows auth hotspot failures"}],
                },
            ),
        ]
        engine = RetrievalEngine()
        engine.ingest_documents(docs)

        evidence = engine.retrieve_evidence(
            "image table risk matrix for auth failure clustering",
            top_k=5,
            diversify=False,
        )

        covered = set(evidence.covered_modalities)
        self.assertIn("table", covered)
        self.assertIn("image", covered)
        self.assertGreater(evidence.aggregate_confidence, 0.4)
        self.assertNotIn("table", evidence.missing_modalities)
        self.assertNotIn("image", evidence.missing_modalities)

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

    def test_retrieve_evidence_builds_source_bundles(self):
        markdown = """
# Auth Requirements
Authentication regressions are release blocking and require triage.

| risk | severity |
| ---- | -------- |
| auth | high     |

![failure matrix](artifacts/failure-matrix.png)
""".strip()

        engine = RetrievalEngine()
        ingestor = MultiSourceIngestor(engine)
        ingestor.ingest_requirements_markdown("req-md-bundle", markdown)

        evidence = engine.retrieve_evidence(
            "authentication risk matrix for test gap triage",
            top_k=6,
            diversify=False,
        )

        self.assertGreaterEqual(len(evidence.source_bundles), 1)
        top_bundle = evidence.source_bundles[0]
        self.assertEqual(top_bundle.source_id, "req-md-bundle")
        self.assertIn("text", top_bundle.modalities)
        self.assertIn("table", top_bundle.modalities)
        self.assertGreater(top_bundle.coverage_ratio, 0.0)

    def test_prompt_includes_source_bundle_summary(self):
        markdown = """
# Release Risks
Missing negative authorization tests increase release risk.

| component | severity |
| --------- | -------- |
| auth      | high     |
""".strip()
        engine = RetrievalEngine()
        ingestor = MultiSourceIngestor(engine)
        ingestor.ingest_requirements_markdown("req-md-prompt", markdown)

        evidence = engine.retrieve_evidence("authorization release risk matrix", top_k=5)
        prompt = build_analysis_prompt(
            question="Where are the highest-risk test gaps?",
            ranked_context=evidence.ranked_chunks,
            source_bundles=evidence.source_bundles,
        )

        self.assertIn("Source bundle summary", prompt)
        self.assertIn("coverage=", prompt)
        self.assertIn("req-md-prompt", prompt)

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


class TestTFIDFEmbeddingProvider(unittest.TestCase):
    def test_tfidf_encodes_texts(self):
        provider = TFIDFEmbeddingProvider()
        texts = [
            "authentication failure causes release blocking",
            "authorization tests missing negative cases",
            "performance benchmark rendering path",
        ]
        vectors = provider.encode(texts)

        self.assertEqual(len(vectors), 3)
        # Each vector should be normalized
        for v in vectors:
            self.assertEqual(len(v), len(provider._vocabulary))
            magnitude = sum(x * x for x in v) ** 0.5
            self.assertAlmostEqual(magnitude, 1.0, places=5)

    def test_tfidf_similarity_ranking(self):
        provider = TFIDFEmbeddingProvider()
        texts = [
            "authentication failure causes release blocking",
            "authorization tests missing negative cases",
            "performance benchmark rendering path",
        ]
        vectors = provider.encode(texts)

        # Query should be most similar to the first text
        query = "authentication failure"
        query_vec = provider.encode([query])[0]

        from src.test_analysis_assistant.retrieval import _cosine_similarity
        similarities = [_cosine_similarity(query_vec, v) for v in vectors]

        # First text should be most similar
        self.assertGreater(similarities[0], similarities[1])
        self.assertGreater(similarities[0], similarities[2])

    def test_tfidf_vocabulary_limits_features(self):
        provider = TFIDFEmbeddingProvider(max_features=10)
        texts = [f"word{i} word{i+1} word{i+2}" for i in range(20)]
        provider.encode(texts)

        self.assertLessEqual(len(provider._vocabulary), 10)

    def test_tfidf_min_df_filters_rare_terms(self):
        provider = TFIDFEmbeddingProvider(min_df=2)
        texts = [
            "uniqueword1 common",
            "uniqueword2 common",
            "common",
        ]
        provider.encode(texts)

        # "uniqueword1" and "uniqueword2" should not be in vocabulary
        self.assertNotIn("uniqueword1", provider._vocabulary)
        self.assertNotIn("uniqueword2", provider._vocabulary)


class TestHybridRetrievalEngine(unittest.TestCase):
    def test_hybrid_engine_with_tfidf(self):
        docs = [
            IngestDocument(
                source_id="req-1",
                source_type=SourceType.REQUIREMENTS,
                content="Authentication failures are release blocking.",
            ),
            IngestDocument(
                source_id="doc-2",
                source_type=SourceType.KNOWLEDGE,
                content="UI rendering optimization for dashboard cards.",
            ),
        ]

        engine = create_hybrid_engine(
            embedding_provider=TFIDFEmbeddingProvider(),
            lexical_weight=0.5,
        )
        engine.ingest_documents(docs)

        ranked = engine.query("release blocking authentication", top_k=2, use_hybrid=True)

        self.assertEqual(len(ranked), 2)
        # First result should be req-1 since it matches "release" and "blocking"
        self.assertEqual(ranked[0].chunk.source_id, "req-1")

    def test_hybrid_engine_lexical_only_mode(self):
        docs = [
            IngestDocument(
                source_id="req-1",
                source_type=SourceType.REQUIREMENTS,
                content="Authentication failures are release blocking.",
            ),
            IngestDocument(
                source_id="doc-2",
                source_type=SourceType.KNOWLEDGE,
                content="UI rendering optimization for dashboard cards.",
            ),
        ]

        engine = create_hybrid_engine(embedding_provider=TFIDFEmbeddingProvider())
        engine.ingest_documents(docs)

        ranked = engine.query("release blocking authentication", top_k=2, use_hybrid=False)

        self.assertEqual(len(ranked), 2)
        # Even with use_hybrid=False, should return results via lexical path
        self.assertGreater(ranked[0].score, 0)

    def test_hybrid_engine_score_breakdown_includes_semantic(self):
        docs = [
            IngestDocument(
                source_id="req-1",
                source_type=SourceType.REQUIREMENTS,
                content="Test failures require P0 triage.",
            ),
        ]

        engine = HybridRetrievalEngine(embedding_provider=TFIDFEmbeddingProvider())
        engine.ingest_documents(docs)

        ranked = engine.query("P0 triage test failures", top_k=1, use_hybrid=True)

        self.assertIn("semantic", ranked[0].score_breakdown)
        self.assertGreater(ranked[0].score_breakdown["semantic"], 0)

    def test_create_hybrid_engine_factory(self):
        engine = create_hybrid_engine(
            chunk_size=200,
            chunk_overlap=20,
            lexical_weight=0.6,
        )

        self.assertIsInstance(engine, HybridRetrievalEngine)
        self.assertEqual(engine._lexical_weight, 0.6)
        self.assertEqual(engine._semantic_weight, 0.4)

    def test_dummy_embedding_provider_still_works(self):
        provider = DummyEmbeddingProvider()
        texts = ["test failure", "authentication error", "release blocking"]

        vectors = provider.encode(texts)

        self.assertEqual(len(vectors), 3)
        # Dummy provider uses 64 dimensions
        self.assertEqual(len(vectors[0]), 64)


if __name__ == "__main__":
    unittest.main()
