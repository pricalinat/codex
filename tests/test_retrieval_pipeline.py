import unittest
import tempfile
from pathlib import Path

from src.test_analysis_assistant.retrieval import (
    AnalysisEvidencePack,
    ArtifactBundle,
    DummyEmbeddingProvider,
    FocusedEvidence,
    HybridRetrievalEngine,
    IngestionRecord,
    IngestDocument,
    MultiSourceIngestor,
    QueryPlan,
    RetrievalEngine,
    SourceType,
    TFIDFEmbeddingProvider,
    build_analysis_prompt,
    build_analysis_prompt_from_evidence,
    build_analysis_prompt_from_pack,
    create_code_aware_engine,
    create_hybrid_engine,
)


class TestRetrievalPipeline(unittest.TestCase):
    def test_retrieve_analysis_evidence_pack_builds_focus_and_merged_results(self):
        docs = [
            IngestDocument(
                source_id="repo-auth",
                source_type=SourceType.REPOSITORY,
                content="Flaky login failures cluster around token refresh traceback paths.",
            ),
            IngestDocument(
                source_id="sys-auth",
                source_type=SourceType.SYSTEM_ANALYSIS,
                content="Root cause hypothesis points to retry storm and timeout cascade in auth service.",
            ),
            IngestDocument(
                source_id="req-auth",
                source_type=SourceType.REQUIREMENTS,
                content="Missing negative credential tests are release risk and require mitigation plan.",
            ),
        ]
        engine = RetrievalEngine()
        engine.ingest_documents(docs)

        pack = engine.retrieve_analysis_evidence_pack("authentication failures in release tests", top_k_per_focus=2)

        self.assertIsInstance(pack, AnalysisEvidencePack)
        self.assertEqual(len(pack.focus_results), 5)
        self.assertTrue(all(isinstance(item, FocusedEvidence) for item in pack.focus_results))
        self.assertGreater(len(pack.merged_evidence.ranked_chunks), 0)
        self.assertGreaterEqual(pack.overall_confidence, 0.0)
        self.assertLessEqual(pack.overall_confidence, 1.0)
        self.assertIn("root_cause", pack.focus_confidence)
        self.assertIn("test_gap", pack.focus_confidence)

    def test_build_analysis_prompt_from_pack_includes_focus_coverage(self):
        docs = [
            IngestDocument(
                source_id="req-auth",
                source_type=SourceType.REQUIREMENTS,
                content="Authentication test gap and release risk prioritization details.",
            ),
            IngestDocument(
                source_id="sys-auth",
                source_type=SourceType.SYSTEM_ANALYSIS,
                content="Failure clustering and root cause notes for auth retries.",
            ),
        ]
        engine = RetrievalEngine()
        engine.ingest_documents(docs)
        pack = engine.retrieve_analysis_evidence_pack("auth release risk", top_k_per_focus=2)

        prompt = build_analysis_prompt_from_pack("Analyze auth failures", pack)

        self.assertIn("Analysis focus coverage", prompt)
        self.assertIn("failure_clustering", prompt)
        self.assertIn("root_cause", prompt)
        self.assertIn("Source bundle summary", prompt)

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

    def test_retrieve_evidence_reports_cross_source_conflict_signal(self):
        docs = [
            IngestDocument(
                source_id="req-auth",
                source_type=SourceType.REQUIREMENTS,
                content="Auth token refresh mitigation is enabled and stable for release.",
            ),
            IngestDocument(
                source_id="sys-auth",
                source_type=SourceType.SYSTEM_ANALYSIS,
                content="Auth token refresh mitigation is disabled and unstable in production.",
            ),
        ]
        engine = RetrievalEngine()
        engine.ingest_documents(docs)

        evidence = engine.retrieve_evidence(
            "auth token refresh mitigation stability",
            top_k=3,
            diversify=False,
            use_expansion=False,
            adaptive_recovery=False,
        )

        self.assertIn("cross_source_conflict", evidence.confidence_factors)
        self.assertGreater(evidence.confidence_factors["cross_source_conflict"], 0.0)

    def test_conflicting_evidence_reduces_calibrated_confidence(self):
        aligned_docs = [
            IngestDocument(
                source_id="req-auth",
                source_type=SourceType.REQUIREMENTS,
                content="Auth retry mitigation is enabled and stable for release readiness.",
            ),
            IngestDocument(
                source_id="sys-auth",
                source_type=SourceType.SYSTEM_ANALYSIS,
                content="Auth retry mitigation is enabled and stable in incident diagnostics.",
            ),
        ]
        conflicting_docs = [
            IngestDocument(
                source_id="req-auth",
                source_type=SourceType.REQUIREMENTS,
                content="Auth retry mitigation is enabled and stable for release readiness.",
            ),
            IngestDocument(
                source_id="sys-auth",
                source_type=SourceType.SYSTEM_ANALYSIS,
                content="Auth retry mitigation is disabled and unstable in incident diagnostics.",
            ),
        ]

        aligned_engine = RetrievalEngine()
        aligned_engine.ingest_documents(aligned_docs)
        aligned = aligned_engine.retrieve_evidence(
            "auth retry mitigation stability",
            top_k=3,
            diversify=False,
            use_expansion=False,
            adaptive_recovery=False,
        )

        conflicting_engine = RetrievalEngine()
        conflicting_engine.ingest_documents(conflicting_docs)
        conflicting = conflicting_engine.retrieve_evidence(
            "auth retry mitigation stability",
            top_k=3,
            diversify=False,
            use_expansion=False,
            adaptive_recovery=False,
        )

        self.assertGreater(
            conflicting.confidence_factors.get("cross_source_conflict", 0.0),
            aligned.confidence_factors.get("cross_source_conflict", 0.0),
        )
        self.assertLess(conflicting.calibrated_confidence, aligned.calibrated_confidence)

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

    def test_query_assigns_artifact_graph_bonus_for_multimodal_linked_source(self):
        docs = [
            IngestDocument(
                source_id="req-single",
                source_type=SourceType.REQUIREMENTS,
                content="Authentication risk matrix requires mitigation planning for release.",
            ),
            IngestDocument(
                source_id="incident-auth",
                source_type=SourceType.SYSTEM_ANALYSIS,
                modality="compound",
                content={
                    "text": "Authentication risk matrix requires mitigation planning for release.",
                    "tables": [{"rows": [{"component": "auth", "risk": "high", "view": "matrix"}]}],
                    "images": [{"ocr_text": "authentication risk heatmap screenshot"}],
                },
            ),
        ]

        engine = RetrievalEngine()
        engine.ingest_documents(docs)

        ranked = engine.query(
            "authentication risk matrix image evidence",
            top_k=4,
            diversify=False,
        )

        self.assertGreaterEqual(len(ranked), 2)
        by_source = {item.chunk.source_id: item for item in ranked}
        self.assertIn("req-single", by_source)
        self.assertIn("incident-auth", by_source)
        self.assertGreater(by_source["incident-auth"].score_breakdown.get("artifact_graph", 0.0), 0.0)
        self.assertGreater(
            by_source["incident-auth"].score_breakdown.get("artifact_graph", 0.0),
            by_source["req-single"].score_breakdown.get("artifact_graph", 0.0),
        )

    def test_retrieve_evidence_reports_artifact_graph_support_factor(self):
        docs = [
            IngestDocument(
                source_id="incident-auth",
                source_type=SourceType.SYSTEM_ANALYSIS,
                modality="compound",
                content={
                    "text": "Authentication risk mitigation plan for release readiness.",
                    "tables": [{"rows": [{"component": "auth", "risk": "high", "priority": "p0"}]}],
                    "images": [{"ocr_text": "authentication risk heatmap for retry storm"}],
                },
            ),
        ]

        engine = RetrievalEngine()
        engine.ingest_documents(docs)
        evidence = engine.retrieve_evidence(
            "auth risk matrix and image evidence",
            top_k=4,
            diversify=False,
            use_expansion=False,
            adaptive_recovery=False,
        )

        self.assertIn("artifact_graph_support", evidence.confidence_factors)
        self.assertGreater(evidence.confidence_factors["artifact_graph_support"], 0.0)

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

    def test_repository_ingestion_marks_code_files_with_code_modality(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "service.py").write_text(
                (
                    "def validate_token(token: str) -> bool:\n"
                    "    return bool(token and token.strip())\n"
                ),
                encoding="utf-8",
            )

            engine = RetrievalEngine()
            ingestor = MultiSourceIngestor(engine)
            chunks = ingestor.ingest_repository(str(root), max_files=10)

            repo_chunks = [chunk for chunk in chunks if chunk.source_id == "repo:service.py"]
            self.assertGreaterEqual(len(repo_chunks), 1)
            self.assertTrue(any(chunk.modality == "code" for chunk in repo_chunks))

    def test_repository_ingestion_emits_repo_manifest_for_structure_retrieval(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "src").mkdir()
            (root / "src" / "auth_token_parser.py").write_text(
                (
                    "def parse_refresh_token(payload):\n"
                    "    return payload.get('refresh_token')\n"
                ),
                encoding="utf-8",
            )
            (root / "docs").mkdir()
            (root / "docs" / "overview.md").write_text(
                "# System Overview\nAuth token parser runs before session validation.",
                encoding="utf-8",
            )

            engine = RetrievalEngine()
            ingestor = MultiSourceIngestor(engine)
            chunks = ingestor.ingest_repository(str(root), max_files=10)

            manifest_chunks = [chunk for chunk in chunks if chunk.metadata.get("manifest_type")]
            self.assertGreaterEqual(len(manifest_chunks), 1)
            self.assertTrue(any(chunk.source_id.startswith("repo:__manifest__") for chunk in manifest_chunks))

            ranked = engine.query("parse_refresh_token file inventory", top_k=3, diversify=False)
            self.assertTrue(any(item.chunk.source_id.startswith("repo:__manifest__") for item in ranked))

    def test_code_aware_repository_ingestion_keeps_repo_manifest_fallback(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "service").mkdir()
            (root / "service" / "auth_handler.py").write_text(
                (
                    "class AuthHandler:\n"
                    "    def validate_session(self, payload):\n"
                    "        return bool(payload)\n"
                ),
                encoding="utf-8",
            )

            engine, ingestor = create_code_aware_engine()
            chunks = ingestor.ingest_repository(str(root), max_files=10)

            manifest_chunks = [chunk for chunk in chunks if chunk.metadata.get("manifest_type")]
            self.assertGreaterEqual(len(manifest_chunks), 1)
            self.assertTrue(any(chunk.source_id.startswith("repo:__manifest__") for chunk in manifest_chunks))

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

    def test_ingest_records_normalizes_multimodal_payloads(self):
        engine = RetrievalEngine()
        ingestor = MultiSourceIngestor(engine)

        chunks = ingestor.ingest_records(
            [
                IngestionRecord(
                    source_id="record:compound",
                    source_type=SourceType.SYSTEM_ANALYSIS,
                    payload={
                        "text": "Auth retry storms impact release stability.",
                        "tables": [{"rows": [{"component": "auth", "risk": "high"}]}],
                        "images": [{"image_path": "screens/auth.png", "alt_text": "auth retry heatmap"}],
                    },
                    metadata={"origin_path": "docs/incidents/auth.json"},
                ),
                IngestionRecord(
                    source_id="record:table",
                    source_type=SourceType.REQUIREMENTS,
                    payload={"rows": [{"requirement": "negative_auth", "status": "missing"}]},
                ),
            ]
        )

        self.assertGreaterEqual(len(chunks), 4)
        self.assertTrue(any(chunk.modality == "text" for chunk in chunks))
        self.assertTrue(any(chunk.modality == "table" for chunk in chunks))
        self.assertTrue(any(chunk.modality == "image_ocr_stub" for chunk in chunks))
        self.assertTrue(all("ingestion_route" in chunk.metadata for chunk in chunks))

    def test_ingest_records_resolves_markdown_file_reference_payload(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            markdown_path = root / "auth_requirements.md"
            markdown_path.write_text(
                (
                    "# Auth Requirements\n"
                    "Missing negative tests are release blocking.\n\n"
                    "| risk | severity |\n"
                    "| ---- | -------- |\n"
                    "| auth | high |\n\n"
                    "![auth heatmap](artifacts/auth-heatmap.png)\n"
                ),
                encoding="utf-8",
            )

            engine = RetrievalEngine()
            ingestor = MultiSourceIngestor(engine)
            chunks = ingestor.ingest_records(
                [
                    IngestionRecord(
                        source_id="record:req-file",
                        source_type=SourceType.REQUIREMENTS,
                        payload={"file_path": str(markdown_path)},
                    )
                ]
            )

            self.assertGreaterEqual(len(chunks), 3)
            self.assertTrue(any(chunk.modality == "text" for chunk in chunks))
            self.assertTrue(any(chunk.modality == "table" for chunk in chunks))
            self.assertTrue(any(chunk.modality == "image_ocr_stub" for chunk in chunks))
            self.assertTrue(all(chunk.metadata.get("ingestion_route") == "file_reference_record" for chunk in chunks))
            self.assertTrue(all(chunk.metadata.get("origin_path") == markdown_path.as_posix() for chunk in chunks))

    def test_ingest_records_resolves_code_file_reference_to_code_modality(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            code_path = root / "auth.py"
            code_path.write_text(
                (
                    "def refresh(token: str) -> str:\n"
                    "    if not token:\n"
                    "        raise ValueError('missing token')\n"
                    "    return token.strip()\n"
                ),
                encoding="utf-8",
            )

            engine = RetrievalEngine()
            ingestor = MultiSourceIngestor(engine)
            chunks = ingestor.ingest_records(
                [
                    IngestionRecord(
                        source_id="record:code-file",
                        source_type=SourceType.CODE_SNIPPET,
                        payload={"path": str(code_path)},
                    )
                ]
            )

            self.assertGreaterEqual(len(chunks), 1)
            self.assertTrue(all(chunk.modality == "code" for chunk in chunks))
            self.assertTrue(all(chunk.metadata.get("origin_path") == code_path.as_posix() for chunk in chunks))

    def test_ingest_records_image_file_reference_uses_sidecar_ocr_when_present(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            image_path = root / "auth-heatmap.png"
            image_path.write_bytes(b"\x89PNG\r\n\x1a\n")
            (root / "auth-heatmap.png.ocr.txt").write_text(
                "retry budget exceeded in auth gateway under load",
                encoding="utf-8",
            )

            engine = RetrievalEngine()
            ingestor = MultiSourceIngestor(engine)
            chunks = ingestor.ingest_records(
                [
                    IngestionRecord(
                        source_id="record:image-sidecar",
                        source_type=SourceType.SYSTEM_ANALYSIS,
                        payload={"file_path": str(image_path)},
                    )
                ]
            )

            self.assertEqual(1, len(chunks))
            image_chunk = chunks[0]
            self.assertEqual("image", image_chunk.modality)
            self.assertIn("retry budget exceeded", image_chunk.text)
            self.assertEqual("pipeline_verified_sidecar", image_chunk.metadata.get("ingestion_route"))
            self.assertTrue(image_chunk.metadata.get("ocr_sidecar_path", "").endswith(".ocr.txt"))

    def test_sidecar_ocr_improves_ingestion_route_quality_vs_stub(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            image_path = root / "auth-graph.png"
            image_path.write_bytes(b"\x89PNG\r\n\x1a\n")
            (root / "auth-graph.png.ocr.txt").write_text(
                "auth retry storm evidence from production screenshot",
                encoding="utf-8",
            )

            sidecar_engine = RetrievalEngine()
            sidecar_ingestor = MultiSourceIngestor(sidecar_engine)
            sidecar_ingestor.ingest_records(
                [
                    IngestionRecord(
                        source_id="record:image-sidecar",
                        source_type=SourceType.SYSTEM_ANALYSIS,
                        payload={"file_path": str(image_path)},
                    )
                ]
            )
            sidecar_evidence = sidecar_engine.retrieve_evidence(
                "auth retry storm screenshot evidence",
                top_k=1,
                diversify=False,
                use_expansion=False,
                adaptive_recovery=False,
            )

            stub_engine = RetrievalEngine()
            stub_ingestor = MultiSourceIngestor(stub_engine)
            stub_ingestor.ingest_records(
                [
                    IngestionRecord(
                        source_id="record:image-stub",
                        source_type=SourceType.SYSTEM_ANALYSIS,
                        payload={"image_path": str(root / "missing.png"), "alt_text": "auth retry storm screenshot"},
                    )
                ]
            )
            stub_evidence = stub_engine.retrieve_evidence(
                "auth retry storm screenshot evidence",
                top_k=1,
                diversify=False,
                use_expansion=False,
                adaptive_recovery=False,
            )

            self.assertGreater(
                sidecar_evidence.confidence_factors["ingestion_route_quality"],
                stub_evidence.confidence_factors["ingestion_route_quality"],
            )

    def test_retrieve_evidence_reports_ingestion_route_quality_factor(self):
        verified_engine = RetrievalEngine()
        verified_ingestor = MultiSourceIngestor(verified_engine)
        verified_ingestor.ingest_records(
            [
                IngestionRecord(
                    source_id="verified:req",
                    source_type=SourceType.REQUIREMENTS,
                    payload="Auth retry heatmap risk matrix requires mitigation.",
                    metadata={"ingestion_route": "pipeline_verified"},
                )
            ]
        )
        verified = verified_engine.retrieve_evidence(
            "auth retry heatmap risk matrix",
            top_k=1,
            diversify=False,
            use_expansion=False,
            adaptive_recovery=False,
        )

        stub_engine = RetrievalEngine()
        stub_ingestor = MultiSourceIngestor(stub_engine)
        stub_ingestor.ingest_records(
            [
                IngestionRecord(
                    source_id="stub:image",
                    source_type=SourceType.SYSTEM_ANALYSIS,
                    payload={"image_path": "screens/auth.png", "alt_text": "auth retry heatmap risk matrix"},
                )
            ]
        )
        stubbed = stub_engine.retrieve_evidence(
            "auth retry heatmap risk matrix",
            top_k=1,
            diversify=False,
            use_expansion=False,
            adaptive_recovery=False,
        )

        self.assertIn("ingestion_route_quality", verified.confidence_factors)
        self.assertIn("ingestion_route_quality", stubbed.confidence_factors)
        self.assertGreater(
            verified.confidence_factors["ingestion_route_quality"],
            stubbed.confidence_factors["ingestion_route_quality"],
        )

    def test_ingest_documents_can_emit_source_summary_chunks(self):
        engine = RetrievalEngine()

        chunks = engine.ingest_documents(
            [
                IngestDocument(
                    source_id="analysis:auth-composite",
                    source_type=SourceType.SYSTEM_ANALYSIS,
                    modality="compound",
                    content={
                        "text": "Auth retry storms correlate with token refresh latency spikes.",
                        "tables": [{"rows": [{"component": "auth", "risk": "high", "signal": "retry_storm"}]}],
                        "images": [{"ocr_text": "heatmap highlights token refresh hotspots"}],
                    },
                )
            ],
            generate_source_summaries=True,
        )

        summary_chunks = [chunk for chunk in chunks if chunk.metadata.get("synthetic_unit") == "source_summary"]
        self.assertEqual(1, len(summary_chunks))
        self.assertEqual("analysis:auth-composite::__summary__", summary_chunks[0].source_id)
        self.assertEqual("text", summary_chunks[0].modality)

    def test_source_summary_chunk_is_retrievable_for_multimodal_bridge_query(self):
        engine = RetrievalEngine()
        engine.ingest_documents(
            [
                IngestDocument(
                    source_id="analysis:bridge",
                    source_type=SourceType.SYSTEM_ANALYSIS,
                    modality="compound",
                    content={
                        "text": "Release risk is elevated for auth under retry storms.",
                        "tables": [{"rows": [{"component": "auth", "priority": "p0", "risk": "high"}]}],
                        "images": [{"ocr_text": "retry storm heatmap in auth gateway"}],
                    },
                ),
            ],
            generate_source_summaries=True,
        )

        ranked = engine.query(
            "source summary auth retry storm release risk p0",
            top_k=3,
            diversify=False,
        )

        self.assertGreaterEqual(len(ranked), 1)
        self.assertEqual("analysis:bridge::__summary__", ranked[0].chunk.source_id)
        self.assertIn("source_summary", ranked[0].chunk.text)

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

    def test_retrieve_evidence_limits_single_source_dominance_when_alternatives_exist(self):
        docs = [
            IngestDocument(
                source_id="req-monolith",
                source_type=SourceType.REQUIREMENTS,
                content="auth risk matrix negative test gap root cause traceback token refresh",
            ),
            IngestDocument(
                source_id="req-monolith",
                source_type=SourceType.REQUIREMENTS,
                content="auth risk matrix negative test gap release blocking root cause",
            ),
            IngestDocument(
                source_id="req-monolith",
                source_type=SourceType.REQUIREMENTS,
                content="auth risk matrix negative test gap mitigation plan root cause",
            ),
            IngestDocument(
                source_id="sys-incident",
                source_type=SourceType.SYSTEM_ANALYSIS,
                content="auth root cause traceback in gateway service diagnostics",
            ),
        ]
        engine = RetrievalEngine()
        engine.ingest_documents(docs)

        evidence = engine.retrieve_evidence(
            "auth root cause traceback risk matrix negative test gap",
            top_k=3,
            diversify=True,
            use_expansion=False,
            adaptive_recovery=False,
        )

        self.assertEqual(3, len(evidence.ranked_chunks))
        source_ids = [item.chunk.source_id for item in evidence.ranked_chunks]
        self.assertIn("sys-incident", source_ids)

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

    def test_artifact_bundle_respects_per_unit_extraction_confidence(self):
        engine = RetrievalEngine()
        ingestor = MultiSourceIngestor(engine)

        chunks = ingestor.ingest_artifact_bundle(
            ArtifactBundle(
                source_id="analysis:confidence-weights",
                source_type=SourceType.SYSTEM_ANALYSIS,
                text="Auth failures spike during token refresh.",
                tables=[
                    {
                        "rows": [{"component": "auth", "risk": "high"}],
                        "extraction_confidence": 0.31,
                    }
                ],
                images=[
                    {
                        "ocr_text": "auth retry heatmap",
                        "extraction_confidence": 0.22,
                    }
                ],
                metadata={"text_extraction_confidence": 0.93},
            )
        )

        self.assertGreaterEqual(len(chunks), 3)
        table_chunk = next(item for item in chunks if item.modality == "table")
        image_chunk = next(item for item in chunks if item.modality == "image")
        text_chunk = next(item for item in chunks if item.modality == "text")

        self.assertAlmostEqual(table_chunk.metadata["extraction_confidence"], 0.31, places=4)
        self.assertAlmostEqual(image_chunk.metadata["extraction_confidence"], 0.22, places=4)
        self.assertAlmostEqual(text_chunk.metadata["extraction_confidence"], 0.93, places=4)

    def test_retrieve_evidence_reports_extraction_reliability_factor(self):
        docs = [
            IngestDocument(
                source_id="sys-ocr-stub",
                source_type=SourceType.SYSTEM_ANALYSIS,
                content={"image_path": "screens/auth.png"},
                modality="image",
            ),
            IngestDocument(
                source_id="req-auth",
                source_type=SourceType.REQUIREMENTS,
                content="Authentication failures require risk triage and mitigation.",
            ),
        ]
        engine = RetrievalEngine()
        engine.ingest_documents(docs)

        evidence = engine.retrieve_evidence(
            "authentication risk triage with image heatmap evidence",
            top_k=3,
            diversify=False,
            use_expansion=False,
            adaptive_recovery=False,
        )

        self.assertIn("extraction_reliability", evidence.confidence_factors)
        self.assertLess(evidence.confidence_factors["extraction_reliability"], 0.9)

    def test_retrieve_evidence_reports_source_reliability_factor(self):
        docs = [
            IngestDocument(
                source_id="sys-ocr-stub",
                source_type=SourceType.SYSTEM_ANALYSIS,
                content={"image_path": "screens/auth.png", "alt_text": "auth retry heatmap risk"},
                modality="image",
            ),
            IngestDocument(
                source_id="req-auth",
                source_type=SourceType.REQUIREMENTS,
                content="Authentication retries are release blocking and need mitigation.",
            ),
        ]
        engine = RetrievalEngine()
        engine.ingest_documents(docs)

        evidence = engine.retrieve_evidence(
            "auth retry heatmap risk mitigation",
            top_k=3,
            diversify=False,
            use_expansion=False,
            adaptive_recovery=False,
        )

        self.assertIn("source_reliability", evidence.confidence_factors)
        self.assertGreaterEqual(evidence.confidence_factors["source_reliability"], 0.0)
        self.assertLessEqual(evidence.confidence_factors["source_reliability"], 1.0)

    def test_prompt_source_bundle_summary_includes_reliability(self):
        docs = [
            IngestDocument(
                source_id="sys-auth",
                source_type=SourceType.SYSTEM_ANALYSIS,
                modality="compound",
                content={
                    "text": "Auth retries fail under load and create release risk.",
                    "tables": [{"rows": [{"component": "auth", "risk": "high"}]}],
                    "images": [{"ocr_text": "auth retry heatmap risk matrix"}],
                },
                metadata={"origin_path": "docs/incidents/auth.md"},
            ),
        ]
        engine = RetrievalEngine()
        engine.ingest_documents(docs)
        evidence = engine.retrieve_evidence("auth retry heatmap risk matrix", top_k=4, diversify=False)

        prompt = build_analysis_prompt(
            question="Summarize risk evidence.",
            ranked_context=evidence.ranked_chunks,
            source_bundles=evidence.source_bundles,
        )

        self.assertIn("reliability=", prompt)

    def test_retrieve_evidence_recommends_ingestion_actions_when_preferred_coverage_missing(self):
        docs = [
            IngestDocument(
                source_id="req-auth",
                source_type=SourceType.REQUIREMENTS,
                content="Authentication risk is release blocking and needs mitigation planning.",
            ),
        ]
        engine = RetrievalEngine()
        engine.ingest_documents(docs)

        evidence = engine.retrieve_evidence(
            "image table root cause traceback evidence for authentication failures",
            top_k=3,
            diversify=False,
            use_expansion=False,
            adaptive_recovery=False,
        )

        self.assertGreaterEqual(len(evidence.recommended_ingestion_actions), 1)
        actions_text = "\n".join(evidence.recommended_ingestion_actions).lower()
        self.assertIn("ocr", actions_text)
        self.assertIn("table", actions_text)
        self.assertIn("system analysis", actions_text)

    def test_prompt_from_evidence_includes_recommended_ingestion_actions(self):
        docs = [
            IngestDocument(
                source_id="req-auth",
                source_type=SourceType.REQUIREMENTS,
                content="Authentication risk is release blocking and needs mitigation planning.",
            ),
        ]
        engine = RetrievalEngine()
        engine.ingest_documents(docs)

        evidence = engine.retrieve_evidence(
            "image table root cause traceback evidence for authentication failures",
            top_k=3,
            diversify=False,
            use_expansion=False,
            adaptive_recovery=False,
        )
        prompt = build_analysis_prompt_from_evidence("Summarize auth risk", evidence)

        self.assertIn("Recommended ingestion actions", prompt)
        self.assertIn("ocr", prompt.lower())


if __name__ == "__main__":
    unittest.main()
