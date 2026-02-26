"""Unified ingestion pipeline for coordinating multimodal and multisource content.

This module provides a unified pipeline that orchestrates different ingestors
for processing various content types and sources:

- Text content (plain text, markdown, documentation)
- Code content (source files, test files, snippets)
- Multimodal content (tables, images with OCR)
- Structured documents (requirements, specs, analysis docs)

The pipeline handles:
- Content type detection and routing
- Parallel processing of different source types
- Error recovery and fallback strategies
- Progress tracking and metadata collection
"""

import hashlib
import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Protocol, Sequence

from .content_detector import (
    ContentCategory,
    DocumentType,
    detect_content_type,
    suggest_ingestion_strategy,
)
from .multimodal import (
    ModalityType,
    OCRResult,
    ProcessedArtifact,
    TableExtractor,
)
from .retrieval import (
    Chunk,
    IngestDocument,
    SourceType,
)

logger = logging.getLogger(__name__)


class IngestionHandler(Protocol):
    """Protocol for ingestion handlers."""

    def can_handle(self, content: Any, metadata: Dict[str, Any]) -> bool:
        """Check if this handler can process the content."""
        ...

    def process(self, content: Any, metadata: Dict[str, Any]) -> "ProcessingResult":
        """Process the content and return chunks."""
        ...


@dataclass
class ProcessingResult:
    """Result of processing content through an ingestion handler."""

    success: bool
    chunks: List[Chunk] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    processing_time_ms: float = 0.0


@dataclass
class SourceConfig:
    """Configuration for a specific source type."""

    source_type: SourceType
    enabled: bool = True
    priority: int = 0  # Higher = more priority
    chunk_size: int = 512
    chunk_overlap: int = 50
    extract_metadata: bool = True
    fallback_enabled: bool = True


@dataclass
class PipelineConfig:
    """Configuration for the unified ingestion pipeline."""

    # Source configurations
    sources: Dict[SourceType, SourceConfig] = field(default_factory=dict)

    # Pipeline behavior
    parallel_processing: bool = True
    max_concurrent: int = 4
    fail_fast: bool = False  # If True, stop on first error

    # Content handling
    extract_tables: bool = True
    extract_images: bool = False  # Disabled by default (requires OCR)
    min_chunk_size: int = 50
    max_chunk_size: int = 2048

    # Metadata
    include_content_hash: bool = True
    include_line_numbers: bool = True

    def __post_init__(self) -> None:
        """Initialize default source configs if not provided."""
        if not self.sources:
            self.sources = {
                SourceType.CODE_SNIPPET: SourceConfig(
                    source_type=SourceType.CODE_SNIPPET,
                    priority=10,
                    chunk_size=384,
                    chunk_overlap=40,
                ),
                SourceType.REPOSITORY: SourceConfig(
                    source_type=SourceType.REPOSITORY,
                    priority=8,
                    chunk_size=512,
                    chunk_overlap=50,
                ),
                SourceType.REQUIREMENTS: SourceConfig(
                    source_type=SourceType.REQUIREMENTS,
                    priority=7,
                    chunk_size=640,
                    chunk_overlap=60,
                ),
                SourceType.SYSTEM_ANALYSIS: SourceConfig(
                    source_type=SourceType.SYSTEM_ANALYSIS,
                    priority=6,
                    chunk_size=640,
                    chunk_overlap=60,
                ),
                SourceType.KNOWLEDGE: SourceConfig(
                    source_type=SourceType.KNOWLEDGE,
                    priority=5,
                    chunk_size=512,
                    chunk_overlap=50,
                ),
            }


@dataclass
class IngestionProgress:
    """Progress tracking for pipeline execution."""

    total_sources: int = 0
    processed_sources: int = 0
    total_documents: int = 0
    processed_documents: int = 0
    total_chunks: int = 0
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def is_complete(self) -> bool:
        return self.processed_sources >= self.total_sources

    @property
    def progress_percent(self) -> float:
        if self.total_sources == 0:
            return 0.0
        return (self.processed_sources / self.total_sources) * 100


@dataclass
class PipelineResult:
    """Result of running the ingestion pipeline."""

    success: bool
    chunks: List[Chunk] = field(default_factory=list)
    source_stats: Dict[SourceType, int] = field(default_factory=dict)
    modality_stats: Dict[str, int] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    progress: IngestionProgress = field(default_factory=IngestionProgress)
    metadata: Dict[str, Any] = field(default_factory=dict)


class UnifiedIngestionPipeline:
    """Unified pipeline for processing multimodal and multisource content.

    This pipeline coordinates different ingestion handlers to process
    various content types efficiently:

    - Detects content type and routes to appropriate handlers
    - Processes content in parallel when possible
    - Provides error recovery and fallback strategies
    - Tracks progress and collects metadata
    """

    def __init__(self, config: Optional[PipelineConfig] = None) -> None:
        """Initialize the unified ingestion pipeline.

        Args:
            config: Pipeline configuration. Uses defaults if not provided.
        """
        self._config = config or PipelineConfig()
        self._table_extractor = TableExtractor()
        self._handlers: List[IngestionHandler] = []
        self._progress = IngestionProgress()

    def register_handler(self, handler: IngestionHandler) -> None:
        """Register a custom ingestion handler.

        Args:
            handler: Handler to register
        """
        self._handlers.append(handler)

    def process_document(self, document: IngestDocument) -> PipelineResult:
        """Process a single document through the pipeline.

        Args:
            document: Document to process

        Returns:
            PipelineResult with processed chunks and metadata
        """
        result = PipelineResult(success=True)
        result.source_stats[document.source_type] = 0
        result.modality_stats[document.modality] = 0

        try:
            # Detect content type
            detection_text = self._build_detection_text(document.content)
            source_hint = str(
                document.metadata.get("file_path")
                or document.metadata.get("path")
                or document.source_id
            )
            content_result = detect_content_type(detection_text, source_hint=source_hint)
            content_category = content_result.category
            recommended_chunker, recommended_source_type = suggest_ingestion_strategy(
                detection_text,
                source_hint=source_hint,
            )

            # Get source config
            source_config = self._config.sources.get(
                document.source_type,
                SourceConfig(source_type=document.source_type),
            )

            # Route to appropriate handler
            # Route based on modality in document metadata
            route_name = "text"
            if document.modality == "compound":
                route_name = "compound"
                chunks = self._process_compound_content(document, source_config)
            elif document.modality == "table" or content_category == ContentCategory.STRUCTURED_DATA:
                route_name = "table"
                chunks = self._process_table_content(document, source_config)
                if not chunks:
                    route_name = "table_fallback_text"
                    chunks = self._process_text_content(document, source_config)
            elif document.modality == "image":
                route_name = "image"
                chunks = self._process_image_content(document, source_config)
            elif (
                document.source_type == SourceType.CODE_SNIPPET
                or recommended_chunker == "code_aware"
            ):
                route_name = "code"
                chunks = self._process_code_content(document, source_config)
            else:
                chunks = self._process_text_content(document, source_config)
            self._attach_detection_metadata(
                chunks=chunks,
                document=document,
                route_name=route_name,
                detection_category=content_result.category.value,
                detection_confidence=content_result.confidence,
                recommended_chunker=recommended_chunker,
                recommended_source_type=recommended_source_type,
                detected_language=(
                    content_result.detected_language.value
                    if content_result.detected_language
                    else ""
                ),
                detected_doc_type=(
                    content_result.detected_doc_type.value
                    if content_result.detected_doc_type
                    else ""
                ),
            )

            result.chunks = chunks
            result.source_stats[document.source_type] = len(chunks)
            result.modality_stats[document.modality] = len(chunks)

            # Add content hash if configured
            if self._config.include_content_hash:
                for chunk in chunks:
                    chunk.metadata["content_hash"] = self._compute_content_hash(
                        chunk.text
                    )

        except Exception as e:
            result.success = False
            result.errors.append(f"Error processing document {document.source_id}: {str(e)}")
            logger.error(f"Error processing document {document.source_id}: {e}")

        return result

    def _build_detection_text(self, content: Any) -> str:
        """Build normalized text used for content-type detection."""
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        if isinstance(content, dict):
            text_parts: List[str] = []
            for key in ("text", "summary", "body"):
                text_value = content.get(key)
                if isinstance(text_value, str):
                    text_parts.append(text_value)
            for key in ("tables", "images", "table", "image"):
                value = content.get(key)
                if value:
                    text_parts.append(str(value))
            if text_parts:
                return "\n".join(text_parts)
        return str(content)

    def _attach_detection_metadata(
        self,
        chunks: Sequence[Chunk],
        document: IngestDocument,
        route_name: str,
        detection_category: str,
        detection_confidence: float,
        recommended_chunker: str,
        recommended_source_type: str,
        detected_language: str,
        detected_doc_type: str,
    ) -> None:
        """Attach detection and routing metadata used by downstream retrieval scoring."""
        for chunk in chunks:
            chunk.metadata.setdefault("ingestion_route", "pipeline_detected")
            chunk.metadata.setdefault("source_id", document.source_id)
            chunk.metadata["processing_handler"] = route_name
            chunk.metadata["detection_category"] = detection_category
            chunk.metadata["detection_confidence"] = round(float(detection_confidence), 4)
            chunk.metadata["recommended_chunker"] = recommended_chunker
            chunk.metadata["recommended_source_type"] = recommended_source_type
            if detected_language:
                chunk.metadata["detected_language"] = detected_language
            if detected_doc_type:
                chunk.metadata["detected_doc_type"] = detected_doc_type

    def process_batch(
        self, documents: Sequence[IngestDocument]
    ) -> PipelineResult:
        """Process multiple documents through the pipeline.

        Args:
            documents: Sequence of documents to process

        Returns:
            Combined PipelineResult with all processed chunks
        """
        # Initialize progress tracking
        self._progress.total_sources = len(set(d.source_type for d in documents))
        self._progress.total_documents = len(documents)

        # Combine results
        combined_result = PipelineResult(success=True)

        # Group by source type for priority processing
        by_source: Dict[SourceType, List[IngestDocument]] = {}
        for doc in documents:
            by_source.setdefault(doc.source_type, []).append(doc)

        # Sort by priority
        sorted_sources = sorted(
            by_source.keys(),
            key=lambda s: self._config.sources.get(s, SourceConfig(s)).priority,
            reverse=True,
        )

        # Process each source type
        for source_type in sorted_sources:
            source_docs = by_source[source_type]

            # Check if source is enabled
            source_config = self._config.sources.get(
                source_type,
                SourceConfig(source_type=source_type),
            )
            if not source_config.enabled:
                combined_result.warnings.append(
                    f"Source {source_type.value} is disabled, skipping"
                )
                continue

            # Process documents for this source
            for doc in source_docs:
                doc_result = self.process_document(doc)

                # Combine chunks
                combined_result.chunks.extend(doc_result.chunks)

                # Combine stats
                for src, count in doc_result.source_stats.items():
                    combined_result.source_stats[src] = (
                        combined_result.source_stats.get(src, 0) + count
                    )
                for mod, count in doc_result.modality_stats.items():
                    combined_result.modality_stats[mod] = (
                        combined_result.modality_stats.get(mod, 0) + count
                    )

                # Track errors and warnings
                combined_result.errors.extend(doc_result.errors)
                combined_result.warnings.extend(doc_result.warnings)

                # Update progress
                self._progress.processed_documents += 1

            self._progress.processed_sources += 1

        # Determine overall success
        combined_result.success = len(combined_result.errors) == 0 or not self._config.fail_fast
        combined_result.progress = self._progress

        return combined_result

    def _process_text_content(
        self, document: IngestDocument, config: SourceConfig
    ) -> List[Chunk]:
        """Process text content.

        Args:
            document: Document to process
            config: Source configuration

        Returns:
            List of chunks
        """
        chunks = []
        content = str(document.content)

        # Split into paragraphs first
        paragraphs = re.split(r"\n\s*\n", content)

        current_chunk = []
        current_size = 0

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            para_size = len(para)

            # Check if paragraph itself is too large
            if para_size > config.chunk_size:
                # Flush current chunk
                if current_chunk:
                    chunks.append(self._create_chunk(
                        " ".join(current_chunk),
                        document,
                        config,
                    ))
                    current_chunk = []
                    current_size = 0

                # Split large paragraph
                chunks.extend(self._split_large_text(para, document, config))
            elif current_size + para_size > config.chunk_size:
                # Flush current chunk
                if current_chunk:
                    chunks.append(self._create_chunk(
                        " ".join(current_chunk),
                        document,
                        config,
                    ))
                current_chunk = [para]
                current_size = para_size
            else:
                current_chunk.append(para)
                current_size += para_size

        # Flush remaining
        if current_chunk:
            chunks.append(self._create_chunk(
                " ".join(current_chunk),
                document,
                config,
            ))

        return chunks

    def _process_code_content(
        self, document: IngestDocument, config: SourceConfig
    ) -> List[Chunk]:
        """Process code content with structure awareness.

        Args:
            document: Document to process
            config: Source configuration

        Returns:
            List of chunks
        """
        from .code_chunker import (
            CodeChunk,
            CodeLanguage,
            detect_language,
            extract_code_units,
        )

        chunks = []
        content = str(document.content)

        # Detect language
        language = detect_language(
            document.metadata.get("file_path", ""),
            content,
        )

        # Extract code units
        code_units = extract_code_units(content, language)

        for unit in code_units:
            if unit.end_line - unit.start_line < 3:
                # Skip very small units
                continue

            chunk_text = f"{unit.name}: {unit.code}"
            if len(chunk_text) < config.chunk_size:
                chunks.append(Chunk(
                    chunk_id=self._generate_chunk_id(document.source_id, unit.start_line),
                    source_id=document.source_id,
                    source_type=document.source_type,
                    modality="code",
                    text=chunk_text,
                    token_count=self._estimate_tokens(chunk_text),
                    metadata={
                        **document.metadata,
                        "language": language.value if language else "unknown",
                        "unit_type": unit.unit_type,
                        "unit_name": unit.name,
                        "start_line": unit.start_line,
                        "end_line": unit.end_line,
                    },
                ))

        # If no units extracted, degrade gracefully with code-labeled chunks.
        if not chunks and content.strip():
            text_chunks = self._split_large_text(content, document, config)
            for chunk in text_chunks:
                chunk.modality = "code"
                chunk.metadata.update(
                    {
                        **document.metadata,
                        "language": language.value if language else "unknown",
                        "unit_type": "snippet_fallback",
                    }
                )
            return text_chunks

        # If content is empty, fall back to text processing
        if not chunks:
            return self._process_text_content(document, config)

        return chunks

    def _process_table_content(
        self, document: IngestDocument, config: SourceConfig
    ) -> List[Chunk]:
        """Process table content.

        Args:
            document: Document to process
            config: Source configuration

        Returns:
            List of chunks
        """
        chunks = []

        # Extract tables from markdown content
        tables = self._table_extractor.extract_from_markdown(str(document.content))

        for table in tables:
            # Convert table to text representation
            rows_str = "\n".join(
                " | ".join(row) for row in [table.headers] + table.rows
            )
            table_text = f"Table: {table.caption or 'Untitled'}\n{rows_str}"

            chunks.append(Chunk(
                chunk_id=self._generate_chunk_id(document.source_id, table.table_id),
                source_id=document.source_id,
                source_type=document.source_type,
                modality="table",
                text=table_text,
                token_count=self._estimate_tokens(table_text),
                metadata={
                    "table_id": table.table_id,
                    "caption": table.caption,
                    "row_count": len(table.rows),
                    "column_count": len(table.headers),
                    "extraction_confidence": table.extraction_confidence,
                },
            ))

        return chunks

    def _process_compound_content(
        self, document: IngestDocument, config: SourceConfig
    ) -> List[Chunk]:
        """Process mixed text/table/image payloads into modality-preserving chunks."""
        if not isinstance(document.content, dict):
            return self._process_text_content(document, config)

        chunks: List[Chunk] = []
        payload = document.content
        bundle_parent_id = f"{document.source_id}::compound"
        component_ids: List[str] = []

        text_parts: List[str] = []
        for key in ("text", "summary", "body"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                text_parts.append(value.strip())
        if text_parts:
            text_doc = IngestDocument(
                source_id=document.source_id,
                source_type=document.source_type,
                content="\n\n".join(text_parts),
                modality="text",
                metadata=dict(document.metadata),
            )
            text_chunks = self._process_text_content(text_doc, config)
            for text_idx, text_chunk in enumerate(text_chunks):
                unit_id = f"text_{text_idx}"
                text_chunk.metadata["unit_kind"] = "text"
                text_chunk.metadata["parent_source_id"] = bundle_parent_id
                text_chunk.metadata["compound_component_id"] = unit_id
                component_ids.append(unit_id)
            chunks.extend(text_chunks)

        table_payloads: List[Any] = []
        if payload.get("table") is not None:
            table_payloads.append(payload.get("table"))
        tables_value = payload.get("tables")
        if isinstance(tables_value, list):
            table_payloads.extend(tables_value)
        elif tables_value is not None:
            table_payloads.append(tables_value)

        for table_idx, table_payload in enumerate(table_payloads):
            table_text = self._table_payload_to_text(table_payload)
            if not table_text.strip():
                continue
            component_id = f"table_{table_idx}"
            component_ids.append(component_id)
            chunks.append(
                Chunk(
                    chunk_id=self._generate_chunk_id(document.source_id, f"compound_table_{table_idx}"),
                    source_id=document.source_id,
                    source_type=document.source_type,
                    modality="table",
                    text=table_text[: config.chunk_size],
                    token_count=self._estimate_tokens(table_text),
                    metadata={
                        **document.metadata,
                        "content_length": len(table_text),
                        "line_count": table_text.count("\n") + 1,
                        "table_index": table_idx,
                        "unit_kind": "table",
                        "parent_source_id": bundle_parent_id,
                        "compound_component_id": component_id,
                    },
                )
            )

        image_payloads: List[Any] = []
        if payload.get("image") is not None:
            image_payloads.append(payload.get("image"))
        images_value = payload.get("images")
        if isinstance(images_value, list):
            image_payloads.extend(images_value)
        elif images_value is not None:
            image_payloads.append(images_value)

        for image_idx, image_payload in enumerate(image_payloads):
            image_doc = IngestDocument(
                source_id=document.source_id,
                source_type=document.source_type,
                content=image_payload,
                modality="image",
                metadata={
                    **document.metadata,
                    "image_index": image_idx,
                },
            )
            image_chunks = self._process_image_content(image_doc, config)
            for chunk_idx, image_chunk in enumerate(image_chunks):
                unit_id = f"image_{image_idx}_{chunk_idx}"
                image_chunk.metadata["parent_source_id"] = bundle_parent_id
                image_chunk.metadata["compound_component_id"] = unit_id
                component_ids.append(unit_id)
            chunks.extend(image_chunks)

        if chunks:
            manifest_chunk = self._build_compound_manifest_chunk(
                document=document,
                config=config,
                parent_source_id=bundle_parent_id,
                component_ids=component_ids,
            )
            if manifest_chunk is not None:
                chunks.append(manifest_chunk)
            return chunks
        return self._process_text_content(document, config)

    def _build_compound_manifest_chunk(
        self,
        document: IngestDocument,
        config: SourceConfig,
        parent_source_id: str,
        component_ids: Sequence[str],
    ) -> Optional[Chunk]:
        """Create a compact manifest chunk that links all compound components."""
        if not component_ids:
            return None

        manifest_lines = [
            f"Compound source {document.source_id} includes {len(component_ids)} components.",
            "Component inventory:",
        ]
        manifest_lines.extend(f"- {component_id}" for component_id in component_ids)
        manifest_text = "\n".join(manifest_lines)

        return Chunk(
            chunk_id=self._generate_chunk_id(document.source_id, "compound_manifest"),
            source_id=parent_source_id,
            source_type=document.source_type,
            modality="text",
            text=manifest_text[: config.chunk_size],
            token_count=self._estimate_tokens(manifest_text),
            metadata={
                **document.metadata,
                "parent_source_id": document.source_id,
                "manifest_type": "compound_parent_manifest",
                "unit_kind": "manifest",
                "component_ids": list(component_ids),
                "content_length": len(manifest_text),
                "line_count": manifest_text.count("\n") + 1,
                "extraction_confidence": 0.9,
            },
        )

    def _table_payload_to_text(self, payload: Any) -> str:
        """Render structured table payloads into retrieval-friendly text."""
        if isinstance(payload, str):
            tables = self._table_extractor.extract_from_markdown(payload)
            if tables:
                rendered: List[str] = []
                for table in tables:
                    lines = [" | ".join(table.headers)] if table.headers else []
                    lines.extend(" | ".join(row) for row in table.rows)
                    rendered.append("\n".join(lines))
                return "\n\n".join(rendered)
            return payload
        if isinstance(payload, dict):
            rows = payload.get("rows")
            if isinstance(rows, list):
                rendered_rows: List[str] = []
                for row in rows:
                    if isinstance(row, dict):
                        rendered_rows.append(", ".join(f"{k}={v}" for k, v in sorted(row.items())))
                    elif isinstance(row, list):
                        rendered_rows.append(" | ".join(str(cell) for cell in row))
                    else:
                        rendered_rows.append(str(row))
                return "\n".join(rendered_rows)
            return str(payload)
        if isinstance(payload, list):
            return "\n".join(self._table_payload_to_text(item) for item in payload)
        return str(payload)

    def _process_image_content(
        self, document: IngestDocument, config: SourceConfig
    ) -> List[Chunk]:
        """Process image content (placeholder for OCR).

        Args:
            document: Document to process
            config: Source configuration

        Returns:
            List of chunks
        """
        image_text = ""
        image_path = ""
        alt_text = ""
        extraction_confidence = 0.3

        if isinstance(document.content, dict):
            ocr_text = str(document.content.get("ocr_text", "")).strip()
            image_path = str(document.content.get("image_path", "")).strip()
            alt_text = str(document.content.get("alt_text", "")).strip()
            if ocr_text:
                image_text = ocr_text
                extraction_confidence = 0.8
            else:
                subject = image_path or "image"
                if alt_text:
                    image_text = f"[OCR_STUB] {subject}: {alt_text}"
                else:
                    image_text = f"[OCR_STUB] no OCR pipeline connected for {subject}"
                extraction_confidence = 0.3 if not self._config.extract_images else 0.5
        else:
            rendered = str(document.content).strip()
            if rendered:
                image_text = rendered
                extraction_confidence = 0.6

        if not image_text:
            return []

        return [
            Chunk(
                chunk_id=self._generate_chunk_id(document.source_id, f"image_{image_path or 'stub'}"),
                source_id=document.source_id,
                source_type=document.source_type,
                modality="image",
                text=image_text[: config.chunk_size],
                token_count=self._estimate_tokens(image_text),
                metadata={
                    **document.metadata,
                    "content_length": len(image_text),
                    "line_count": image_text.count("\n") + 1,
                    "unit_kind": "image",
                    "image_path": image_path,
                    "alt_text": alt_text,
                    "extraction_confidence": round(extraction_confidence, 4),
                },
            )
        ]

    def _split_large_text(
        self, text: str, document: IngestDocument, config: SourceConfig
    ) -> List[Chunk]:
        """Split large text into smaller chunks.

        Args:
            text: Text to split
            document: Source document
            config: Source configuration

        Returns:
            List of chunks
        """
        chunks = []
        lines = text.split("\n")

        current_lines = []
        current_size = 0

        for i, line in enumerate(lines):
            line_size = len(line)
            if current_size + line_size > config.chunk_size and current_lines:
                chunks.append(self._create_chunk(
                    "\n".join(current_lines),
                    document,
                    config,
                ))
                current_lines = []
                current_size = 0

            current_lines.append(line)
            current_size += line_size

        if current_lines:
            chunks.append(self._create_chunk(
                "\n".join(current_lines),
                document,
                config,
            ))

        return chunks

    def _create_chunk(
        self, text: str, document: IngestDocument, config: SourceConfig
    ) -> Chunk:
        """Create a chunk from text.

        Args:
            text: Chunk text
            document: Source document
            config: Source configuration

        Returns:
            Chunk
        """
        # Trim to max size if needed
        if len(text) > config.chunk_size:
            text = text[:config.chunk_size]

        return Chunk(
            chunk_id=self._generate_chunk_id(document.source_id, len(text)),
            source_id=document.source_id,
            source_type=document.source_type,
            modality=document.modality,
            text=text,
            token_count=self._estimate_tokens(text),
            metadata={
                "content_length": len(text),
                "line_count": text.count("\n") + 1,
            },
        )

    def _generate_chunk_id(self, source_id: str, index: Any) -> str:
        """Generate a unique chunk ID.

        Args:
            source_id: Source identifier
            index: Index or position

        Returns:
            Unique chunk ID
        """
        return f"{source_id}_{index}"

    def _compute_content_hash(self, text: str) -> str:
        """Compute hash of content for deduplication.

        Args:
            text: Text to hash

        Returns:
            SHA256 hash hex string
        """
        return hashlib.sha256(text.encode()).hexdigest()[:16]

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count for text.

        Args:
            text: Text to count tokens for

        Returns:
            Estimated token count
        """
        # Rough estimate: ~4 chars per token
        return len(text) // 4


def create_unified_pipeline(
    sources: Optional[Dict[SourceType, SourceConfig]] = None,
    **kwargs,
) -> UnifiedIngestionPipeline:
    """Create a unified ingestion pipeline with default or custom config.

    Args:
        sources: Optional source configurations
        **kwargs: Additional pipeline config options

    Returns:
        Configured UnifiedIngestionPipeline
    """
    config = PipelineConfig(sources=sources, **kwargs)
    return UnifiedIngestionPipeline(config)
