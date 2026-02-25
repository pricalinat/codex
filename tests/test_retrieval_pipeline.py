import unittest
import tempfile
from pathlib import Path

from src.test_analysis_assistant.retrieval import (
    ArtifactBundle,
    DummyEmbeddingProvider,
    HybridRetrievalEngine,
    IngestDocument,
    MultiSourceIngestor,
    QueryPlan,
    RetrievalEngine,
    SourceType,
    TFIDFEmbeddingProvider,
    build_analysis_prompt,
    build_analysis_prompt_from_evidence,
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

    def test_retrieve_evidence_reports_unavailable_corpus_coverage(self):
        docs = [
            IngestDocument(
                source_id="req-auth",
                source_type=SourceType.REQUIREMENTS,
                content="Authentication release risk and missing test coverage details.",
            ),
        ]
        engine = RetrievalEngine()
        engine.ingest_documents(docs)

        evidence = engine.retrieve_evidence(
            "image table risk matrix for authentication gaps",
            top_k=3,
            diversify=False,
            use_expansion=False,
            adaptive_recovery=False,
        )

        self.assertIn("table", evidence.unavailable_preferred_modalities)
        self.assertIn("image", evidence.unavailable_preferred_modalities)
        self.assertIn("image_ocr_stub", evidence.unavailable_preferred_modalities)
        self.assertTrue(len(evidence.unavailable_preferred_source_types) > 0)
        self.assertEqual(
            set(evidence.unavailable_preferred_source_types).intersection(set(evidence.covered_source_types)),
            set(),
        )

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

    def test_retrieve_evidence_calibrates_confidence_for_missing_coverage(self):
        docs = [
            IngestDocument(
                source_id="req-auth",
                source_type=SourceType.REQUIREMENTS,
                content="Authentication risk requires release prioritization and mitigation plan.",
            ),
        ]
        engine = RetrievalEngine()
        engine.ingest_documents(docs)

        evidence = engine.retrieve_evidence(
            "image table risk matrix for authentication gaps",
            top_k=3,
            diversify=False,
            use_expansion=False,
            adaptive_recovery=False,
        )

        self.assertIn("source_coverage", evidence.confidence_factors)
        self.assertIn("modality_coverage", evidence.confidence_factors)
        self.assertIn("ocr_stub_ratio", evidence.confidence_factors)
        self.assertIn("cross_source_consensus", evidence.confidence_factors)
        self.assertIn("source_concentration", evidence.confidence_factors)
        self.assertLess(evidence.calibrated_confidence, evidence.aggregate_confidence)
        self.assertLess(evidence.confidence_factors["modality_coverage"], 1.0)

    def test_retrieve_evidence_calibration_stays_close_with_rich_multimodal_coverage(self):
        docs = [
            IngestDocument(
                source_id="compound-rich",
                source_type=SourceType.SYSTEM_ANALYSIS,
                modality="compound",
                content={
                    "text": "Auth release risk matrix identifies retry storms and priority mitigations.",
                    "tables": [{"rows": [{"component": "auth", "risk": "high", "priority": "p0"}]}],
                    "images": [{"ocr_text": "auth retry heatmap with risk clusters"}],
                },
            ),
        ]
        engine = RetrievalEngine()
        engine.ingest_documents(docs)

        evidence = engine.retrieve_evidence(
            "image table risk matrix for auth retry storms",
            top_k=5,
            diversify=False,
            use_expansion=False,
            adaptive_recovery=False,
        )

        self.assertGreaterEqual(evidence.confidence_factors["modality_coverage"], 0.75)
        self.assertGreaterEqual(
            evidence.calibrated_confidence,
            evidence.aggregate_confidence * 0.7,
        )

    def test_retrieve_evidence_reports_cross_source_consensus_signal(self):
        docs = [
            IngestDocument(
                source_id="req-auth",
                source_type=SourceType.REQUIREMENTS,
                content="Auth token refresh failure is release blocking and requires mitigation.",
            ),
            IngestDocument(
                source_id="sys-auth",
                source_type=SourceType.SYSTEM_ANALYSIS,
                content="Incident diagnostics confirm auth token refresh failure and mitigation urgency.",
            ),
        ]
        engine = RetrievalEngine()
        engine.ingest_documents(docs)

        evidence = engine.retrieve_evidence(
            "auth token refresh failure mitigation",
            top_k=3,
            diversify=False,
            use_expansion=False,
            adaptive_recovery=False,
        )

        self.assertIn("cross_source_consensus", evidence.confidence_factors)
        self.assertGreater(evidence.confidence_factors["cross_source_consensus"], 0.0)

    def test_retrieve_evidence_reports_source_concentration_signal(self):
        docs = [
            IngestDocument(
                source_id="repo-auth",
                source_type=SourceType.REPOSITORY,
                content=(
                    "Auth token refresh failure mitigation with traceback and root cause hypothesis "
                    "for retry storm under load."
                ),
            ),
            IngestDocument(
                source_id="repo-auth",
                source_type=SourceType.REPOSITORY,
                content=(
                    "Additional auth token refresh failure mitigation notes and retry storm analysis."
                ),
            ),
            IngestDocument(
                source_id="kb-noise",
                source_type=SourceType.KNOWLEDGE,
                content="Governance updates and release policy communication templates.",
            ),
        ]
        engine = RetrievalEngine()
        engine.ingest_documents(docs)

        evidence = engine.retrieve_evidence(
            "auth token refresh failure mitigation",
            top_k=3,
            diversify=False,
            use_expansion=False,
            adaptive_recovery=False,
        )

        self.assertIn("source_concentration", evidence.confidence_factors)
        self.assertGreaterEqual(evidence.confidence_factors["source_concentration"], 0.6)

    def test_retrieve_evidence_adaptive_recovery_adds_missing_modalities(self):
        docs = [
            IngestDocument(
                source_id="req-core",
                source_type=SourceType.REQUIREMENTS,
                content=(
                    "Authentication release risk requires prioritization and mitigation plan for failures."
                ),
            ),
            IngestDocument(
                source_id="sys-table",
                source_type=SourceType.SYSTEM_ANALYSIS,
                content={"rows": [{"component": "auth", "risk": "high", "evidence": "matrix"}]},
                modality="table",
            ),
            IngestDocument(
                source_id="sys-image",
                source_type=SourceType.SYSTEM_ANALYSIS,
                content={"ocr_text": "auth failure heatmap from screenshot"},
                modality="image",
            ),
        ]
        engine = RetrievalEngine()
        engine.ingest_documents(docs)

        baseline = engine.retrieve_evidence(
            "image table risk matrix for auth gaps",
            top_k=1,
            diversify=False,
            use_expansion=False,
            adaptive_recovery=False,
        )
        adaptive = engine.retrieve_evidence(
            "image table risk matrix for auth gaps",
            top_k=3,
            diversify=False,
            use_expansion=False,
            adaptive_recovery=True,
        )

        self.assertIn("image", baseline.missing_modalities)
        self.assertIn("table", {chunk.chunk.modality for chunk in adaptive.ranked_chunks})
        self.assertNotIn("image", adaptive.missing_modalities)
        self.assertTrue(adaptive.recovery_applied)
        self.assertGreater(len(adaptive.recovery_queries), 0)

    def test_retrieve_evidence_adaptive_recovery_tracks_strategy_metadata(self):
        docs = [
            IngestDocument(
                source_id="req-plan",
                source_type=SourceType.REQUIREMENTS,
                content="Release risk triage plan for authentication regressions.",
            ),
        ]
        engine = RetrievalEngine()
        engine.ingest_documents(docs)

        evidence = engine.retrieve_evidence(
            "image table risk matrix for auth",
            top_k=2,
            diversify=False,
            use_expansion=False,
            adaptive_recovery=True,
        )

        self.assertIn(evidence.retrieval_strategy, {"baseline", "adaptive_recovery"})
        if evidence.recovery_applied:
            self.assertEqual(evidence.retrieval_strategy, "adaptive_recovery")
            self.assertGreaterEqual(len(evidence.recovery_queries), 1)

    def test_query_exposes_corroboration_score_component(self):
        docs = [
            IngestDocument(
                source_id="req-auth",
                source_type=SourceType.REQUIREMENTS,
                content="Authentication token refresh failure requires mitigation plan.",
            ),
            IngestDocument(
                source_id="sys-auth",
                source_type=SourceType.SYSTEM_ANALYSIS,
                content="Incident diagnostics confirm token refresh failure in production.",
            ),
            IngestDocument(
                source_id="kb-isolated",
                source_type=SourceType.KNOWLEDGE,
                content="Playbook for release governance and compliance review.",
            ),
        ]
        engine = RetrievalEngine()
        engine.ingest_documents(docs)

        ranked = engine.query("token refresh failure mitigation", top_k=3, diversify=False)

        self.assertGreaterEqual(len(ranked), 2)
        for item in ranked:
            self.assertIn("corroboration", item.score_breakdown)

    def test_query_assigns_higher_corroboration_to_multisource_supported_chunk(self):
        docs = [
            IngestDocument(
                source_id="req-auth",
                source_type=SourceType.REQUIREMENTS,
                content="Token refresh failure needs auth mitigation and test coverage updates.",
            ),
            IngestDocument(
                source_id="sys-auth",
                source_type=SourceType.SYSTEM_ANALYSIS,
                content="Auth incident review confirms token refresh failure and mitigation workstream.",
            ),
            IngestDocument(
                source_id="kb-noise",
                source_type=SourceType.KNOWLEDGE,
                content="Mitigation playbook for governance reporting and audit readiness.",
            ),
        ]
        engine = RetrievalEngine()
        engine.ingest_documents(docs)

        ranked = engine.query("token refresh failure mitigation", top_k=3, diversify=False)
        by_source = {item.chunk.source_id: item for item in ranked}

        self.assertIn("req-auth", by_source)
        self.assertIn("kb-noise", by_source)
        self.assertGreater(
            by_source["req-auth"].score_breakdown.get("corroboration", 0.0),
            by_source["kb-noise"].score_breakdown.get("corroboration", 0.0),
        )

    def test_query_diversify_prefers_intent_source_type_coverage(self):
        docs = [
            IngestDocument(
                source_id="repo-auth-parser",
                source_type=SourceType.REPOSITORY,
                content=(
                    "Root cause traceback shows auth parser failure and hypothesis for token decode path."
                ),
            ),
            IngestDocument(
                source_id="repo-auth-handler",
                source_type=SourceType.REPOSITORY,
                content=(
                    "Root cause traceback indicates handler failure, hypothesis around request normalization."
                ),
            ),
            IngestDocument(
                source_id="sys-incident-auth",
                source_type=SourceType.SYSTEM_ANALYSIS,
                content=(
                    "Incident diagnostics provide root cause hypothesis and traceback chain for auth failures."
                ),
            ),
        ]
        engine = RetrievalEngine()
        engine.ingest_documents(docs)

        ranked = engine.query(
            "root cause traceback hypothesis why auth failures",
            top_k=2,
            diversify=True,
        )

        self.assertEqual(len(ranked), 2)
        source_types = {item.chunk.source_type for item in ranked}
        self.assertIn(SourceType.SYSTEM_ANALYSIS, source_types)

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

    def test_repository_ingestion_preserves_markdown_modalities(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "README.md").write_text(
                (
                    "# Release Risks\n"
                    "Missing negative auth tests are release blocking.\n\n"
                    "| risk | severity |\n"
                    "| ---- | -------- |\n"
                    "| auth | high |\n\n"
                    "![auth matrix](artifacts/auth-matrix.png)\n"
                ),
                encoding="utf-8",
            )

            engine = RetrievalEngine()
            ingestor = MultiSourceIngestor(engine)
            chunks = ingestor.ingest_repository(str(root), max_files=10)

            self.assertGreaterEqual(len(chunks), 3)
            modalities = {chunk.modality for chunk in chunks}
            self.assertIn("text", modalities)
            self.assertIn("table", modalities)
            self.assertIn("image_ocr_stub", modalities)

    def test_repository_ingestion_parses_csv_as_table_modality(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "risk-matrix.csv").write_text(
                "component,severity,evidence\nauth,high,token expiry gaps\n",
                encoding="utf-8",
            )

            engine = RetrievalEngine()
            ingestor = MultiSourceIngestor(engine)
            chunks = ingestor.ingest_repository(str(root), max_files=10)

            self.assertGreaterEqual(len(chunks), 1)
            self.assertTrue(any(chunk.modality == "table" for chunk in chunks))
            table_text = "\n".join(chunk.text for chunk in chunks if chunk.modality == "table")
            self.assertIn("component=auth", table_text)
            self.assertIn("severity=high", table_text)

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

    def test_markdown_table_units_are_normalized_into_key_value_rows(self):
        markdown = """
# Coverage Matrix
| component | severity | owner |
| --------- | -------- | ----- |
| auth      | high     | qa    |
| billing   | medium   | sdet  |
""".strip()

        engine = RetrievalEngine()
        ingestor = MultiSourceIngestor(engine)
        chunks = ingestor.ingest_requirements_markdown("req-md-table-normalized", markdown)

        table_text = "\n".join(chunk.text for chunk in chunks if chunk.modality == "table")
        self.assertIn("component=auth", table_text)
        self.assertIn("severity=high", table_text)
        self.assertIn("owner=qa", table_text)

    def test_markdown_image_alt_text_is_searchable_with_ocr_stub(self):
        markdown = """
# Auth Diagnostics
![auth retry storm heatmap](artifacts/auth-heatmap.png)
""".strip()

        engine = RetrievalEngine()
        ingestor = MultiSourceIngestor(engine)
        ingestor.ingest_requirements_markdown("req-md-image-alt", markdown)

        ranked = engine.query("auth retry storm heatmap", top_k=1, diversify=False)

        self.assertEqual(len(ranked), 1)
        self.assertEqual(ranked[0].chunk.source_id, "req-md-image-alt")
        self.assertEqual(ranked[0].chunk.modality, "image_ocr_stub")
        self.assertIn("auth", ranked[0].matched_terms)
        self.assertIn("heatmap", ranked[0].chunk.text)

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

    def test_ingest_artifact_bundle_creates_multimodal_chunks_with_provenance(self):
        engine = RetrievalEngine()
        ingestor = MultiSourceIngestor(engine)

        chunks = ingestor.ingest_artifact_bundle(
            ArtifactBundle(
                source_id="analysis:auth-incident",
                source_type=SourceType.SYSTEM_ANALYSIS,
                text="Authentication failures spike during token refresh under high concurrency.",
                tables=[
                    {"rows": [{"component": "auth", "risk": "high", "owner": "qa"}]},
                ],
                images=[
                    {"image_path": "screens/auth-heatmap.png", "alt_text": "auth failure heatmap"},
                ],
                metadata={"origin_path": "docs/incidents/auth.md", "page_number": 4},
            )
        )

        self.assertGreaterEqual(len(chunks), 3)
        modalities = {chunk.modality for chunk in chunks}
        self.assertIn("text", modalities)
        self.assertIn("table", modalities)
        self.assertIn("image_ocr_stub", modalities)
        self.assertTrue(all(chunk.metadata.get("origin_path") == "docs/incidents/auth.md" for chunk in chunks))
        self.assertTrue(all(chunk.metadata.get("page_number") == 4 for chunk in chunks))

    def test_artifact_bundle_image_alt_text_is_retrievable(self):
        engine = RetrievalEngine()
        ingestor = MultiSourceIngestor(engine)

        ingestor.ingest_artifact_bundle(
            ArtifactBundle(
                source_id="analysis:auth-visual",
                source_type=SourceType.SYSTEM_ANALYSIS,
                images=[{"image_path": "screens/auth-heatmap.png", "alt_text": "retry storm heatmap"}],
            )
        )

        ranked = engine.query("retry storm heatmap", top_k=1, diversify=False)

        self.assertEqual(1, len(ranked))
        self.assertEqual("analysis:auth-visual", ranked[0].chunk.source_id)
        self.assertEqual("image_ocr_stub", ranked[0].chunk.modality)

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

    def test_retrieve_evidence_prefers_intent_source_coverage_from_candidate_pool(self):
        docs = [
            IngestDocument(
                source_id="repo-auth-a",
                source_type=SourceType.REPOSITORY,
                content="traceback root cause hypothesis auth parser failure in decode path",
            ),
            IngestDocument(
                source_id="repo-auth-b",
                source_type=SourceType.REPOSITORY,
                content="traceback root cause hypothesis auth handler failure in request flow",
            ),
            IngestDocument(
                source_id="sys-auth-incident",
                source_type=SourceType.SYSTEM_ANALYSIS,
                content="incident diagnostics root cause auth failure and traceback chain",
            ),
        ]
        engine = RetrievalEngine()
        engine.ingest_documents(docs)

        evidence = engine.retrieve_evidence(
            "traceback root cause hypothesis why auth failures",
            top_k=2,
            diversify=False,
            use_expansion=False,
            adaptive_recovery=False,
        )

        self.assertEqual(2, len(evidence.ranked_chunks))
        source_types = {item.chunk.source_type for item in evidence.ranked_chunks}
        self.assertIn(SourceType.SYSTEM_ANALYSIS, source_types)

    def test_build_analysis_prompt_from_evidence_includes_confidence_and_missing_signals(self):
        docs = [
            IngestDocument(
                source_id="req-auth",
                source_type=SourceType.REQUIREMENTS,
                content="Authentication release risk requires mitigation and test gap review.",
            ),
        ]
        engine = RetrievalEngine()
        engine.ingest_documents(docs)

        evidence = engine.retrieve_evidence(
            "image table risk matrix for authentication gaps",
            top_k=2,
            diversify=False,
            use_expansion=False,
            adaptive_recovery=False,
        )
        prompt = build_analysis_prompt_from_evidence(
            question="Prioritize test gaps for this release.",
            evidence=evidence,
        )

        self.assertIn("Retrieval confidence:", prompt)
        self.assertIn("Missing retrieval evidence:", prompt)
        self.assertIn("Prioritize test gaps for this release.", prompt)

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

    def test_query_exposes_enhanced_confidence_components(self):
        docs = [
            IngestDocument(
                source_id="req-auth",
                source_type=SourceType.REQUIREMENTS,
                content="Authentication failure mitigation and release risk triage.",
            ),
            IngestDocument(
                source_id="kb-auth",
                source_type=SourceType.KNOWLEDGE,
                content="Authentication failure mitigation and release risk triage.",
            ),
        ]
        engine = RetrievalEngine()
        engine.ingest_documents(docs)

        ranked = engine.query("authentication failure mitigation release risk", top_k=2, diversify=False)

        self.assertEqual(2, len(ranked))
        for item in ranked:
            self.assertIn("authority", item.score_breakdown)
            self.assertIn("position", item.score_breakdown)
            self.assertIn("completeness", item.score_breakdown)

    def test_query_confidence_prefers_authoritative_source_on_tie(self):
        docs = [
            IngestDocument(
                source_id="req-auth-tie",
                source_type=SourceType.REQUIREMENTS,
                content="Authentication failure mitigation release risk triage",
            ),
            IngestDocument(
                source_id="kb-auth-tie",
                source_type=SourceType.KNOWLEDGE,
                content="Authentication failure mitigation release risk triage",
            ),
        ]
        engine = RetrievalEngine()
        engine.ingest_documents(docs)

        ranked = engine.query("authentication failure mitigation release risk triage", top_k=2, diversify=False)
        by_source = {item.chunk.source_id: item for item in ranked}

        self.assertGreater(
            by_source["req-auth-tie"].confidence,
            by_source["kb-auth-tie"].confidence,
        )

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

    def test_tfidf_query_encoding_reuses_fitted_vocabulary(self):
        provider = TFIDFEmbeddingProvider()
        provider.encode(
            [
                "authentication failure causes release blocking",
                "authorization tests missing negative cases",
            ]
        )
        fitted_vocab = dict(provider._vocabulary)
        fitted_size = len(fitted_vocab)

        query_vec = provider.encode(["authentication failure"])[0]

        self.assertEqual(len(query_vec), fitted_size)
        self.assertEqual(provider._vocabulary, fitted_vocab)


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

    def test_hybrid_query_diversify_prefers_intent_source_type_coverage(self):
        docs = [
            IngestDocument(
                source_id="repo-auth-a",
                source_type=SourceType.REPOSITORY,
                content=(
                    "Root cause traceback hypothesis shows auth parser failure in token refresh flow."
                ),
            ),
            IngestDocument(
                source_id="repo-auth-b",
                source_type=SourceType.REPOSITORY,
                content=(
                    "Root cause traceback hypothesis indicates auth handler failure in refresh flow."
                ),
            ),
            IngestDocument(
                source_id="sys-incident-auth",
                source_type=SourceType.SYSTEM_ANALYSIS,
                content=(
                    "System diagnostics report root cause hypothesis with traceback evidence for auth failures."
                ),
            ),
        ]
        engine = create_hybrid_engine(embedding_provider=TFIDFEmbeddingProvider(), lexical_weight=0.9)
        engine.ingest_documents(docs)

        ranked = engine.query(
            "root cause traceback hypothesis why auth failures",
            top_k=2,
            diversify=True,
            use_hybrid=True,
        )

        self.assertEqual(len(ranked), 2)
        source_types = {item.chunk.source_type for item in ranked}
        self.assertIn(SourceType.SYSTEM_ANALYSIS, source_types)

    def test_hybrid_query_does_not_rebuild_tfidf_vocabulary(self):
        docs = [
            IngestDocument(
                source_id="req-auth",
                source_type=SourceType.REQUIREMENTS,
                content="Authentication failures are release blocking and require mitigation.",
            ),
            IngestDocument(
                source_id="sys-auth",
                source_type=SourceType.SYSTEM_ANALYSIS,
                content="Incident diagnostics show token refresh failures under load.",
            ),
        ]
        provider = TFIDFEmbeddingProvider()
        engine = create_hybrid_engine(embedding_provider=provider, lexical_weight=0.35)
        engine.ingest_documents(docs)
        before = dict(provider._vocabulary)

        ranked = engine.query("authentication token refresh failure mitigation", top_k=2, use_hybrid=True)
        after = dict(provider._vocabulary)

        self.assertEqual(before, after)
        self.assertEqual(len(ranked), 2)


if __name__ == "__main__":
    unittest.main()
