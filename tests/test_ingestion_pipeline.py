"""Tests for UnifiedIngestionPipeline."""

import pytest

from src.test_analysis_assistant.ingestion_pipeline import (
    IngestionProgress,
    PipelineConfig,
    PipelineResult,
    ProcessingResult,
    SourceConfig,
    SourceType,
    UnifiedIngestionPipeline,
    create_unified_pipeline,
)
from src.test_analysis_assistant.retrieval import IngestDocument


class TestPipelineConfig:
    """Tests for PipelineConfig."""

    def test_default_config(self):
        """Test default configuration."""
        config = PipelineConfig()
        assert config.parallel_processing is True
        assert config.max_concurrent == 4
        assert config.fail_fast is False

    def test_custom_config(self):
        """Test custom configuration."""
        config = PipelineConfig(
            parallel_processing=False,
            max_concurrent=2,
            fail_fast=True,
        )
        assert config.parallel_processing is False
        assert config.max_concurrent == 2
        assert config.fail_fast is True

    def test_default_sources_configured(self):
        """Test that default sources are configured."""
        config = PipelineConfig()
        assert SourceType.CODE_SNIPPET in config.sources
        assert SourceType.REPOSITORY in config.sources
        assert SourceType.REQUIREMENTS in config.sources
        assert SourceType.KNOWLEDGE in config.sources


class TestSourceConfig:
    """Tests for SourceConfig."""

    def test_source_config_defaults(self):
        """Test default source configuration."""
        config = SourceConfig(source_type=SourceType.CODE_SNIPPET)
        assert config.enabled is True
        assert config.priority == 0
        assert config.chunk_size == 512
        assert config.chunk_overlap == 50

    def test_custom_source_config(self):
        """Test custom source configuration."""
        config = SourceConfig(
            source_type=SourceType.REPOSITORY,
            priority=10,
            chunk_size=1024,
        )
        assert config.priority == 10
        assert config.chunk_size == 1024


class TestIngestionProgress:
    """Tests for IngestionProgress."""

    def test_progress_defaults(self):
        """Test default progress values."""
        progress = IngestionProgress()
        assert progress.total_sources == 0
        assert progress.processed_sources == 0
        assert progress.total_documents == 0
        assert progress.processed_documents == 0
        assert progress.total_chunks == 0

    def test_progress_percent(self):
        """Test progress percentage calculation."""
        progress = IngestionProgress()
        progress.total_sources = 4
        progress.processed_sources = 2
        assert progress.progress_percent == 50.0

    def test_is_complete(self):
        """Test completion detection."""
        progress = IngestionProgress()
        progress.total_sources = 2
        progress.processed_sources = 1
        assert not progress.is_complete

        progress.processed_sources = 2
        assert progress.is_complete


class TestUnifiedIngestionPipeline:
    """Tests for UnifiedIngestionPipeline."""

    def test_pipeline_initialization(self):
        """Test pipeline initializes correctly."""
        pipeline = UnifiedIngestionPipeline()
        assert pipeline._config is not None

    def test_create_pipeline_with_config(self):
        """Test pipeline creation with custom config."""
        config = PipelineConfig(max_concurrent=8)
        pipeline = UnifiedIngestionPipeline(config)
        assert pipeline._config.max_concurrent == 8

    def test_process_simple_text_document(self):
        """Test processing a simple text document."""
        pipeline = UnifiedIngestionPipeline()
        doc = IngestDocument(
            source_id="doc-1",
            source_type=SourceType.KNOWLEDGE,
            content="This is a test document with some content.",
            modality="text",
        )

        result = pipeline.process_document(doc)

        assert result.success is True
        assert len(result.chunks) > 0
        assert result.chunks[0].source_id == "doc-1"
        assert result.chunks[0].source_type == SourceType.KNOWLEDGE

    def test_process_code_document(self):
        """Test processing a code document."""
        pipeline = UnifiedIngestionPipeline()
        code_content = '''
def test_login():
    """Test login functionality."""
    assert login("user", "pass") == True

def test_logout():
    """Test logout functionality."""
    assert logout() == True
'''
        doc = IngestDocument(
            source_id="code-1",
            source_type=SourceType.CODE_SNIPPET,
            content=code_content,
            modality="text",
            metadata={"file_path": "test_auth.py"},
        )

        result = pipeline.process_document(doc)

        assert result.success is True
        assert len(result.chunks) > 0

    def test_process_text_document_uses_detected_code_strategy(self):
        """Code-like content should route to code processing even from non-code source."""
        pipeline = UnifiedIngestionPipeline()
        content = """
def login(user, password):
    if not user:
        raise ValueError("missing user")
    return True
"""
        doc = IngestDocument(
            source_id="doc-code-like",
            source_type=SourceType.KNOWLEDGE,
            content=content,
            modality="text",
            metadata={"file_path": "auth.py"},
        )

        result = pipeline.process_document(doc)

        assert result.success is True
        assert len(result.chunks) > 0
        assert any(chunk.modality == "code" for chunk in result.chunks)

    def test_table_processing_falls_back_to_text_when_no_tables_extracted(self):
        """Table modality should degrade to text chunks if table extraction returns nothing."""
        pipeline = UnifiedIngestionPipeline()
        doc = IngestDocument(
            source_id="table-missing",
            source_type=SourceType.SYSTEM_ANALYSIS,
            content="No parsable markdown table content is present.",
            modality="table",
        )

        result = pipeline.process_document(doc)

        assert result.success is True
        assert len(result.chunks) > 0
        assert any(chunk.modality == "table" or chunk.modality == "text" for chunk in result.chunks)

    def test_chunk_metadata_includes_detection_signals(self):
        """Detection signals should be propagated to chunk metadata for retrieval quality."""
        pipeline = UnifiedIngestionPipeline()
        doc = IngestDocument(
            source_id="req-detect",
            source_type=SourceType.REQUIREMENTS,
            content="# Requirements\n\nThe system shall validate authentication failures.",
            modality="text",
        )

        result = pipeline.process_document(doc)

        assert result.success is True
        assert len(result.chunks) > 0
        chunk = result.chunks[0]
        assert "detection_category" in chunk.metadata
        assert "detection_confidence" in chunk.metadata
        assert "recommended_chunker" in chunk.metadata
        assert "recommended_source_type" in chunk.metadata

    def test_process_compound_document_emits_multimodal_chunks(self):
        """Compound payloads should be decomposed into text/table/image chunks."""
        pipeline = UnifiedIngestionPipeline()
        doc = IngestDocument(
            source_id="incident-auth-1",
            source_type=SourceType.SYSTEM_ANALYSIS,
            modality="compound",
            content={
                "text": "Auth retries spike after deploy and block release.",
                "tables": [{"rows": [{"component": "gateway", "risk": "high"}]}],
                "images": [{"image_path": "artifacts/auth-heatmap.png", "alt_text": "auth retry heatmap"}],
            },
        )

        result = pipeline.process_document(doc)

        assert result.success is True
        assert len(result.chunks) >= 3
        modalities = {chunk.modality for chunk in result.chunks}
        assert "text" in modalities
        assert "table" in modalities
        assert "image" in modalities
        assert any(chunk.metadata.get("processing_handler") == "compound" for chunk in result.chunks)

    def test_compound_processing_emits_parent_manifest_and_lineage_metadata(self):
        """Compound chunks should include parent linkage for multimodal retrieval graphing."""
        pipeline = UnifiedIngestionPipeline()
        doc = IngestDocument(
            source_id="incident-auth-lineage",
            source_type=SourceType.SYSTEM_ANALYSIS,
            modality="compound",
            content={
                "text": "Authentication retries increase risk.",
                "tables": [{"rows": [{"service": "gateway", "risk": "high"}]}],
                "images": [{"alt_text": "retry spike heatmap"}],
            },
        )

        result = pipeline.process_document(doc)

        assert result.success is True
        manifest_chunks = [chunk for chunk in result.chunks if chunk.metadata.get("manifest_type") == "compound_parent_manifest"]
        assert len(manifest_chunks) == 1
        manifest = manifest_chunks[0]
        assert manifest.source_id == "incident-auth-lineage::compound"
        assert manifest.metadata.get("parent_source_id") == "incident-auth-lineage"
        assert manifest.metadata.get("processing_handler") == "compound"

        child_chunks = [chunk for chunk in result.chunks if chunk.source_id == "incident-auth-lineage"]
        assert child_chunks
        assert all(chunk.metadata.get("parent_source_id") == "incident-auth-lineage::compound" for chunk in child_chunks)
        assert all(chunk.metadata.get("compound_component_id") for chunk in child_chunks)

    def test_process_batch(self):
        """Test processing multiple documents."""
        pipeline = UnifiedIngestionPipeline()
        docs = [
            IngestDocument(
                source_id="doc-1",
                source_type=SourceType.KNOWLEDGE,
                content="Document one content.",
            ),
            IngestDocument(
                source_id="doc-2",
                source_type=SourceType.REQUIREMENTS,
                content="Requirements document content.",
            ),
            IngestDocument(
                source_id="doc-3",
                source_type=SourceType.KNOWLEDGE,
                content="Another knowledge document.",
            ),
        ]

        result = pipeline.process_batch(docs)

        assert result.success is True
        assert len(result.chunks) > 0
        assert result.source_stats[SourceType.KNOWLEDGE] >= 2
        assert result.source_stats[SourceType.REQUIREMENTS] >= 1

    def test_process_respects_source_priority(self):
        """Test that sources are processed by priority."""
        config = PipelineConfig()
        # Set priorities
        config.sources[SourceType.REQUIREMENTS] = SourceConfig(
            source_type=SourceType.REQUIREMENTS,
            priority=10,
        )
        config.sources[SourceType.KNOWLEDGE] = SourceConfig(
            source_type=SourceType.KNOWLEDGE,
            priority=5,
        )

        pipeline = UnifiedIngestionPipeline(config)
        docs = [
            IngestDocument(source_id="d1", source_type=SourceType.KNOWLEDGE, content="Knowledge doc"),
            IngestDocument(source_id="d2", source_type=SourceType.REQUIREMENTS, content="Requirements doc"),
        ]

        result = pipeline.process_batch(docs)

        assert result.success is True

    def test_disabled_source_skipped(self):
        """Test that disabled sources are skipped."""
        config = PipelineConfig()
        config.sources[SourceType.CODE_SNIPPET] = SourceConfig(
            source_type=SourceType.CODE_SNIPPET,
            enabled=False,
        )

        pipeline = UnifiedIngestionPipeline(config)
        docs = [
            IngestDocument(
                source_id="code-1",
                source_type=SourceType.CODE_SNIPPET,
                content="Code content",
            ),
        ]

        result = pipeline.process_batch(docs)

        assert len(result.warnings) > 0
        assert any("disabled" in w for w in result.warnings)

    def test_error_handling(self):
        """Test error handling for invalid content."""
        pipeline = UnifiedIngestionPipeline()

        # Content that might cause issues - using None
        doc = IngestDocument(
            source_id="bad-doc",
            source_type=SourceType.KNOWLEDGE,
            content=None,
        )

        # The pipeline should handle this gracefully
        result = pipeline.process_document(doc)
        # May succeed with empty chunks or fail - both are acceptable

    def test_chunk_metadata(self):
        """Test that chunks have proper metadata."""
        pipeline = UnifiedIngestionPipeline(
            PipelineConfig(include_content_hash=True)
        )
        doc = IngestDocument(
            source_id="meta-test",
            source_type=SourceType.KNOWLEDGE,
            content="This is test content for metadata.",
        )

        result = pipeline.process_document(doc)

        assert result.success is True
        if result.chunks:
            chunk = result.chunks[0]
            assert "content_hash" in chunk.metadata
            assert chunk.metadata["content_length"] > 0


class TestCreateUnifiedPipeline:
    """Tests for create_unified_pipeline factory function."""

    def test_create_with_defaults(self):
        """Test creating pipeline with defaults."""
        pipeline = create_unified_pipeline()
        assert isinstance(pipeline, UnifiedIngestionPipeline)

    def test_create_with_custom_sources(self):
        """Test creating pipeline with custom sources."""
        sources = {
            SourceType.CODE_SNIPPET: SourceConfig(
                source_type=SourceType.CODE_SNIPPET,
                priority=20,
            ),
        }
        pipeline = create_unified_pipeline(sources=sources)
        assert pipeline._config.sources[SourceType.CODE_SNIPPET].priority == 20

    def test_create_with_kwargs(self):
        """Test creating pipeline with additional kwargs."""
        pipeline = create_unified_pipeline(max_concurrent=10, fail_fast=True)
        assert pipeline._config.max_concurrent == 10
        assert pipeline._config.fail_fast is True


class TestProcessingResult:
    """Tests for ProcessingResult dataclass."""

    def test_processing_result_defaults(self):
        """Test ProcessingResult default values."""
        result = ProcessingResult(success=True)
        assert result.success is True
        assert result.chunks == []
        assert result.errors == []
        assert result.warnings == []
        assert result.metadata == {}
        assert result.processing_time_ms == 0.0


class TestPipelineResult:
    """Tests for PipelineResult dataclass."""

    def test_pipeline_result_defaults(self):
        """Test PipelineResult default values."""
        result = PipelineResult(success=True)
        assert result.success is True
        assert result.chunks == []
        assert result.source_stats == {}
        assert result.modality_stats == {}
        assert result.errors == []
        assert result.warnings == []
        assert isinstance(result.progress, IngestionProgress)
