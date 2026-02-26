"""Multimodal ingestion support for processing mixed content (tables, images).

This module provides stubs and interfaces for:
- Table extraction from various document formats
- Image OCR processing
- Combined multimodal content ingestion
"""

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence


class ModalityType(str, Enum):
    """Types of content modalities."""

    TEXT = "text"
    TABLE = "table"
    IMAGE = "image"
    COMPOUND = "compound"


@dataclass
class ExtractedTable:
    """Extracted table data with metadata."""

    table_id: str
    headers: List[str]
    rows: List[List[str]]
    caption: Optional[str] = None
    source_location: Optional[str] = None
    extraction_confidence: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OCRResult:
    """OCR processing result for images."""

    image_id: str
    text: str
    bounding_boxes: List[Dict[str, Any]] = field(default_factory=list)
    language: str = "en"
    ocr_confidence: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProcessedArtifact:
    """Processed artifact with extracted content from all modalities."""

    source_id: str
    source_type: str
    text_chunks: List[str] = field(default_factory=list)
    table_data: List[ExtractedTable] = field(default_factory=list)
    ocr_results: List[OCRResult] = field(default_factory=list)
    processing_errors: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class TableExtractor:
    """Extracts tables from various document formats.

    This is a stub implementation that provides basic table extraction.
    In production, this would integrate with libraries like:
    - tabula-py for PDFs
    - openpyxl for Excel
    - python-docx for Word documents
    """

    SUPPORTED_FORMATS = {"markdown", "csv", "html", "excel", "pdf", "docx"}

    def extract_from_markdown(self, content: str) -> List[ExtractedTable]:
        """Extract tables from markdown content.

        Args:
            content: Markdown text potentially containing tables

        Returns:
            List of extracted tables
        """
        tables = []
        lines = content.split("\n")
        in_table = False
        headers: List[str] = []
        rows: List[List[str]] = []
        table_id = 0

        for line in lines:
            stripped = line.strip()
            # Check for table markdown format: | header | header |
            if stripped.startswith("|") and "|" in stripped[1:]:
                parts = [p.strip() for p in stripped.split("|")[1:-1]]
                # Check if it's a separator line (all dashes and/or colons)
                if all(self._is_separator(p) for p in parts):
                    # This is just the separator, skip
                    continue
                if not in_table:
                    # First row is headers
                    headers = parts
                    in_table = True
                else:
                    rows.append(parts)
            elif in_table and not stripped.startswith("|"):
                # End of table
                if headers and rows:
                    tables.append(ExtractedTable(
                        table_id=f"md_table_{table_id}",
                        headers=headers,
                        rows=rows,
                        extraction_confidence=0.85,
                    ))
                    table_id += 1
                headers = []
                rows = []
                in_table = False

        # Handle table at end of content
        if headers and rows:
            tables.append(ExtractedTable(
                table_id=f"md_table_{table_id}",
                headers=headers,
                rows=rows,
                extraction_confidence=0.85,
            ))

        return tables

    def _is_separator(self, cell: str) -> bool:
        """Check if a cell is a markdown table separator."""
        return bool(re.match(r"^[-:]+$", cell))

    def extract_from_csv(self, content: str) -> List[ExtractedTable]:
        """Extract tables from CSV content.

        Args:
            content: CSV text

        Returns:
            List of extracted tables (usually just one)
        """
        import io
        import csv

        tables = []
        try:
            reader = csv.reader(io.StringIO(content))
            rows = list(reader)
            if rows:
                # First row as headers, rest as data
                tables.append(ExtractedTable(
                    table_id="csv_table_0",
                    headers=rows[0],
                    rows=rows[1:] if len(rows) > 1 else [],
                    extraction_confidence=0.95,
                ))
        except Exception:
            pass

        return tables

    def extract_from_json(self, content: str) -> List[ExtractedTable]:
        """Extract tables from JSON content (array of objects or nested).

        Args:
            content: JSON text potentially containing tabular data

        Returns:
            List of extracted tables
        """
        import json

        tables = []
        try:
            data = json.loads(content)

            # Handle array of objects (common API response format)
            if isinstance(data, list) and data and isinstance(data[0], dict):
                if data:  # Non-empty array
                    # Extract headers from all objects
                    headers = list(data[0].keys()) if data else []
                    rows = []
                    for item in data:
                        row = [str(item.get(h, "")) for h in headers]
                        rows.append(row)

                    if headers and rows:
                        tables.append(ExtractedTable(
                            table_id="json_table_0",
                            headers=headers,
                            rows=rows,
                            extraction_confidence=0.9,
                            metadata={"format": "array_of_objects"},
                        ))

            # Handle object with array property (common API response)
            elif isinstance(data, dict):
                for key, value in data.items():
                    if isinstance(value, list) and value and isinstance(value[0], dict):
                        headers = list(value[0].keys()) if value else []
                        rows = []
                        for item in value:
                            row = [str(item.get(h, "")) for h in headers]
                            rows.append(row)

                        if headers and rows:
                            tables.append(ExtractedTable(
                                table_id=f"json_{key}_0",
                                headers=headers,
                                rows=rows,
                                extraction_confidence=0.85,
                                metadata={"format": "nested_array", "source_key": key},
                            ))

        except (json.JSONDecodeError, TypeError):
            pass

        return tables

    def extract_from_delimited(self, content: str, delimiter: str = ",") -> List[ExtractedTable]:
        """Extract tables from delimited text (CSV, TSV, semicolon-separated).

        Args:
            content: Delimited text content
            delimiter: Field delimiter (default: comma)

        Returns:
            List of extracted tables
        """
        import io
        import csv

        tables = []
        try:
            reader = csv.reader(io.StringIO(content), delimiter=delimiter)
            rows = list(reader)
            if rows:
                # Filter out empty rows
                rows = [r for r in rows if any(cell.strip() for cell in r)]
                if rows:
                    # First row as headers, rest as data
                    tables.append(ExtractedTable(
                        table_id=f"delimited_table_0",
                        headers=rows[0],
                        rows=rows[1:] if len(rows) > 1 else [],
                        extraction_confidence=0.95,
                        metadata={"format": f"delimited_{repr(delimiter)}"},
                    ))
        except Exception:
            pass

        return tables

    def extract_from_html(self, content: str) -> List[ExtractedTable]:
        """Extract tables from HTML content.

        Args:
            content: HTML text potentially containing tables

        Returns:
            List of extracted tables
        """
        tables = []
        table_id = 0

        # Simple regex-based table extraction (stub)
        # Production would use BeautifulSoup or lxml
        table_pattern = r"<table[^>]*>(.*?)</table>"
        for match in re.finditer(table_pattern, content, re.DOTALL | re.IGNORECASE):
            table_content = match.group(1)

            # Extract headers
            header_pattern = r"<th[^>]*>(.*?)</th>"
            headers = [re.sub(r"<[^>]+>", "", h).strip() for h in re.findall(header_pattern, table_content, re.DOTALL)]

            # Extract data rows (only td, not th)
            row_pattern = r"<tr[^>]*>(.*?)</tr>"
            rows = []
            for row_match in re.finditer(row_pattern, table_content, re.DOTALL):
                row_content = row_match.group(1)
                # Only look for td cells, not th cells (which are headers)
                cell_pattern = r"<td[^>]*>(.*?)</td>"
                cells = [re.sub(r"<[^>]+>", "", c).strip() for c in re.findall(cell_pattern, row_content, re.DOTALL)]
                if cells:
                    rows.append(cells)

            if headers or rows:
                # If we have headers, rows already contains only data rows
                # (header row had th, not td, so wasn't added)
                tables.append(ExtractedTable(
                    table_id=f"html_table_{table_id}",
                    headers=headers if headers else rows[0] if rows else [],
                    rows=rows if not headers else rows,  # Already filtered
                    source_location="html",
                    extraction_confidence=0.75,
                ))
                table_id += 1

        return tables

    def extract(self, content: str, format_hint: str = "auto") -> List[ExtractedTable]:
        """Extract tables from content with automatic format detection.

        Args:
            content: Content containing tables
            format_hint: Format hint (markdown, csv, html, json, tsv, auto)

        Returns:
            List of extracted tables
        """
        if format_hint == "auto":
            # Auto-detect format
            content_stripped = content.strip()
            if content_stripped.startswith("|"):
                format_hint = "markdown"
            elif content_stripped.startswith("<"):
                format_hint = "html"
            elif content_stripped.startswith(("{" , "[")):
                format_hint = "json"
            elif "\t" in content:
                format_hint = "tsv"
            elif ";" in content and "," not in content:
                # Semicolon-separated (European CSV)
                format_hint = "semicolon"
            elif "," in content:
                format_hint = "csv"

        if format_hint == "markdown":
            return self.extract_from_markdown(content)
        elif format_hint == "csv":
            return self.extract_from_csv(content)
        elif format_hint == "html":
            return self.extract_from_html(content)
        elif format_hint == "json":
            return self.extract_from_json(content)
        elif format_hint == "tsv":
            return self.extract_from_delimited(content, delimiter="\t")
        elif format_hint == "semicolon":
            return self.extract_from_delimited(content, delimiter=";")

        return []


class ImageOCRProcessor:
    """Processes images for OCR extraction.

    This is a stub implementation that provides the interface for OCR.
    In production, this would integrate with:
    - pytesseract (Tesseract OCR)
    - Google Cloud Vision API
    - AWS Textract
    - Azure Computer Vision
    """

    def __init__(self, language: str = "eng") -> None:
        """Initialize OCR processor.

        Args:
            language: Language code for OCR (default: eng)
        """
        self._language = language

    def process_image_bytes(self, image_data: bytes, image_id: str = "image") -> OCRResult:
        """Process image from bytes.

        Args:
            image_data: Raw image bytes
            image_id: Identifier for the image

        Returns:
            OCR result with extracted text
        """
        # Stub: In production, this would call actual OCR
        return OCRResult(
            image_id=image_id,
            text="[OCR stub: image content would be extracted here]",
            language=self._language,
            ocr_confidence=0.0,
            metadata={"stub": True, "note": "Requires OCR library integration"},
        )

    def process_image_path(self, image_path: str) -> OCRResult:
        """Process image from file path.

        Args:
            image_path: Path to image file

        Returns:
            OCR result with extracted text
        """
        # Stub: Would read file and call process_image_bytes
        return OCRResult(
            image_id=image_path,
            text="[OCR stub: image content would be extracted here]",
            language=self._language,
            ocr_confidence=0.0,
            metadata={"stub": True, "path": image_path},
        )

    def process_image_url(self, image_url: str) -> OCRResult:
        """Process image from URL.

        Args:
            image_url: URL to image

        Returns:
            OCR result with extracted text
        """
        # Stub: Would fetch and process
        return OCRResult(
            image_id=image_url,
            text="[OCR stub: image content would be extracted here]",
            language=self._language,
            ocr_confidence=0.0,
            metadata={"stub": True, "url": image_url},
        )


class TesseractOCRProcessor(ImageOCRProcessor):
    """Production-grade OCR processor using Tesseract.

    This processor uses pytesseract for actual OCR text extraction from images.
    Falls back gracefully if pytesseract or Tesseract OCR engine is not available.

    Supported features:
    - Multiple language support (eng, spa, fra, deu, etc.)
    - Configurable PSM (page segmentation mode)
    - Confidence scores for extracted text
    - Bounding box extraction for text regions
    """

    # Common language codes supported by Tesseract
    SUPPORTED_LANGUAGES = {
        "eng": "English",
        "spa": "Spanish",
        "fra": "French",
        "deu": "German",
        "ita": "Italian",
        "por": "Portuguese",
        "rus": "Russian",
        "jpn": "Japanese",
        "kor": "Korean",
        "chi_sim": "Chinese (Simplified)",
        "chi_tra": "Chinese (Traditional)",
    }

    def __init__(
        self,
        language: str = "eng",
        psm: int = 3,
        oem: int = 3,
    ) -> None:
        """Initialize Tesseract OCR processor.

        Args:
            language: Language code for OCR (default: eng)
            psm: Page segmentation mode (default: 3 = Fully automatic)
            oem: OCR engine mode (default: 3 = Default, using LSTM only)
        """
        super().__init__(language=language)
        self._psm = psm
        self._oem = oem
        self._is_available = False
        self._load_tesseract()

    def _load_tesseract(self) -> None:
        """Load pytesseract and verify Tesseract is available."""
        try:
            import pytesseract
            # Verify Tesseract is actually installed
            version = pytesseract.get_tesseract_version()
            self._is_available = True
        except ImportError:
            self._is_available = False
            import sys
            print(
                "Warning: pytesseract not installed. Run: pip install pytesseract",
                file=sys.stderr
            )
        except Exception:
            self._is_available = False
            import sys
            print(
                "Warning: Tesseract OCR engine not found. "
                "Install Tesseract: https://github.com/tesseract-ocr/tesseract",
                file=sys.stderr
            )

    @property
    def is_available(self) -> bool:
        """Check if Tesseract OCR is available."""
        return self._is_available

    @property
    def supported_languages(self) -> Dict[str, str]:
        """Get supported language codes and names."""
        return self.SUPPORTED_LANGUAGES

    def process_image_bytes(self, image_data: bytes, image_id: str = "image") -> OCRResult:
        """Process image from bytes.

        Args:
            image_data: Raw image bytes
            image_id: Identifier for the image

        Returns:
            OCR result with extracted text
        """
        if not self._is_available:
            return super().process_image_bytes(image_data, image_id)

        try:
            import pytesseract
            from PIL import Image
            import io

            # Load image from bytes
            image = Image.open(io.BytesIO(image_data))

            # Configure OCR
            config = f"--psm {self._psm} --oem {self._oem}"

            # Extract text
            text = pytesseract.image_to_string(
                image,
                lang=self._language,
                config=config
            )

            # Get confidence data
            data = pytesseract.image_to_data(
                image,
                lang=self._language,
                config=config,
                output_type=pytesseract.Output.DICT
            )

            # Calculate average confidence
            confidences = [
                int(conf) for conf in data.get("conf", [])
                if conf != "-1"  # -1 means not recognized
            ]
            avg_confidence = sum(confidences) / len(confidences) / 100.0 if confidences else 0.0

            # Extract bounding boxes
            bounding_boxes = []
            n_boxes = len(data["text"])
            for i in range(n_boxes):
                if int(data["conf"][i]) > 0:  # Only include recognized text
                    bounding_boxes.append({
                        "text": data["text"][i],
                        "confidence": float(data["conf"][i]) / 100.0,
                        "left": data["left"][i],
                        "top": data["top"][i],
                        "width": data["width"][i],
                        "height": data["height"][i],
                    })

            return OCRResult(
                image_id=image_id,
                text=text.strip(),
                language=self._language,
                ocr_confidence=avg_confidence,
                bounding_boxes=bounding_boxes,
                metadata={"processor": "tesseract", "psm": self._psm, "oem": self._oem},
            )

        except ImportError as e:
            import sys
            print(f"Warning: Missing dependency for OCR: {e}", file=sys.stderr)
            return OCRResult(
                image_id=image_id,
                text="[OCR failed: Missing PIL or pytesseract]",
                language=self._language,
                ocr_confidence=0.0,
            )
        except Exception as e:
            import sys
            print(f"Warning: OCR processing failed: {e}", file=sys.stderr)
            return OCRResult(
                image_id=image_id,
                text=f"[OCR failed: {str(e)}]",
                language=self._language,
                ocr_confidence=0.0,
            )

    def process_image_path(self, image_path: str) -> OCRResult:
        """Process image from file path.

        Args:
            image_path: Path to image file

        Returns:
            OCR result with extracted text
        """
        if not self._is_available:
            return super().process_image_path(image_path)

        try:
            import pytesseract
            from PIL import Image

            # Load image
            image = Image.open(image_path)

            # Configure OCR
            config = f"--psm {self._psm} --oem {self._oem}"

            # Extract text
            text = pytesseract.image_to_string(
                image,
                lang=self._language,
                config=config
            )

            # Get confidence
            data = pytesseract.image_to_data(
                image,
                lang=self._language,
                config=config,
                output_type=pytesseract.Output.DICT
            )

            confidences = [
                int(conf) for conf in data.get("conf", [])
                if conf != "-1"
            ]
            avg_confidence = sum(confidences) / len(confidences) / 100.0 if confidences else 0.0

            return OCRResult(
                image_id=image_path,
                text=text.strip(),
                language=self._language,
                ocr_confidence=avg_confidence,
                metadata={"processor": "tesseract", "path": image_path, "psm": self._psm},
            )

        except Exception as e:
            import sys
            print(f"Warning: OCR processing failed: {e}", file=sys.stderr)
            return OCRResult(
                image_id=image_path,
                text=f"[OCR failed: {str(e)}]",
                language=self._language,
                ocr_confidence=0.0,
            )


class PDFProcessor:
    """Processes PDF documents for text extraction.

    This is a stub implementation with fallback for PDF text extraction.
    In production, this would integrate with:
    - pdfminer.six (pure Python, recommended)
    - PyMuPDF (fitz) - fast C++ binding
    - pdfplumber - table extraction focus

    The stub provides helpful guidance when PDF processing is needed.
    """

    def __init__(self) -> None:
        """Initialize PDF processor."""
        self._pdfminer_available = self._check_pdfminer()

    def _check_pdfminer(self) -> bool:
        """Check if pdfminer.six is available."""
        try:
            import pdfminer.high_level
            return True
        except ImportError:
            return False

    def extract_text_from_bytes(self, pdf_data: bytes) -> str:
        """Extract text from PDF bytes.

        Args:
            pdf_data: PDF file content as bytes

        Returns:
            Extracted text content
        """
        if self._pdfminer_available:
            import io
            from pdfminer.high_level import extract_text
            return extract_text(io.BytesIO(pdf_data))
        else:
            return self._stub_response("bytes")

    def extract_text_from_path(self, pdf_path: str) -> str:
        """Extract text from PDF file path.

        Args:
            pdf_path: Path to PDF file

        Returns:
            Extracted text content
        """
        if self._pdfminer_available:
            from pdfminer.high_level import extract_text
            return extract_text(pdf_path)
        else:
            return self._stub_response("path")

    def extract_text_from_url(self, pdf_url: str) -> str:
        """Extract text from PDF URL.

        Args:
            pdf_url: URL to PDF file

        Returns:
            Extracted text content
        """
        # For URL extraction, we'd need to fetch the PDF first
        if self._pdfminer_available:
            try:
                import urllib.request
                import io
                from pdfminer.high_level import extract_text
                with urllib.request.urlopen(pdf_url) as response:
                    pdf_data = response.read()
                    return extract_text(io.BytesIO(pdf_data))
            except Exception:
                pass
        return self._stub_response("url")

    def _stub_response(self, source_type: str) -> str:
        """Return a stub response with helpful guidance."""
        return (
            f"[PDF extraction not available: pdfminer.six not installed]\n"
            f"To enable PDF text extraction, install pdfminer.six:\n"
            f"    pip install pdfminer.six\n"
            f"\n"
            f"Source type: {source_type}\n"
            f"\n"
            f"For production use with tables, consider:\n"
            f"    pip install pdfplumber  # Best for table extraction\n"
        )

    @property
    def is_available(self) -> bool:
        """Check if PDF processing is available."""
        return self._pdfminer_available


class MultimodalIngestor:
    """Processes mixed-modality content for ingestion into RAG pipeline.

    Combines text extraction, table extraction, and OCR processing
    to produce normalized chunks suitable for the retrieval engine.
    """

    def __init__(
        self,
        table_extractor: Optional[TableExtractor] = None,
        ocr_processor: Optional[ImageOCRProcessor] = None,
    ) -> None:
        """Initialize multimodal ingestor.

        Args:
            table_extractor: Table extraction handler
            ocr_processor: OCR processing handler
        """
        self._table_extractor = table_extractor or TableExtractor()
        self._ocr_processor = ocr_processor or ImageOCRProcessor()

    def process_artifact_bundle(
        self,
        artifact: "ArtifactBundle",
        include_tables: bool = True,
        include_images: bool = True,
    ) -> ProcessedArtifact:
        """Process an artifact bundle into normalized chunks.

        Args:
            artifact: The artifact bundle to process
            include_tables: Whether to extract tables
            include_images: Whether to process images

        Returns:
            Processed artifact with extracted content
        """
        result = ProcessedArtifact(
            source_id=artifact.source_id,
            source_type=artifact.source_type.value,
        )

        # Process text content
        if artifact.text:
            # Split text into chunks
            text_chunks = self._chunk_text(artifact.text)
            result.text_chunks.extend(text_chunks)

            # Extract tables from text if requested
            if include_tables:
                tables = self._table_extractor.extract(artifact.text)
                result.table_data.extend(tables)

        # Process images if requested
        if include_images and artifact.images:
            for idx, image_data in enumerate(artifact.images):
                if isinstance(image_data, bytes):
                    ocr_result = self._ocr_processor.process_image_bytes(
                        image_data,
                        f"{artifact.source_id}_img_{idx}",
                    )
                elif isinstance(image_data, str):
                    # Could be path or URL
                    if image_data.startswith(("http://", "https://")):
                        ocr_result = self._ocr_processor.process_image_url(image_data)
                    else:
                        ocr_result = self._ocr_processor.process_image_path(image_data)
                else:
                    ocr_result = OCRResult(
                        image_id=f"{artifact.source_id}_img_{idx}",
                        text="[Unsupported image format]",
                        ocr_confidence=0.0,
                    )
                result.ocr_results.append(ocr_result)

        # Store metadata
        result.metadata = dict(artifact.metadata)
        result.metadata["modality"] = self._determine_modality(result)

        return result

    def _chunk_text(self, text: str, max_chunk_size: int = 500) -> List[str]:
        """Split text into chunks.

        Args:
            text: Text to chunk
            max_chunk_size: Maximum characters per chunk

        Returns:
            List of text chunks
        """
        if len(text) <= max_chunk_size:
            return [text] if text.strip() else []

        chunks = []
        paragraphs = text.split("\n\n")

        current_chunk = ""
        for para in paragraphs:
            # If single paragraph is too long, chunk by characters
            if len(para) > max_chunk_size:
                # First flush existing chunk
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
                    current_chunk = ""
                # Chunk the long paragraph
                for i in range(0, len(para), max_chunk_size):
                    chunks.append(para[i:i + max_chunk_size])
            elif len(current_chunk) + len(para) + 2 <= max_chunk_size:
                current_chunk += para + "\n\n"
            else:
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
                current_chunk = para + "\n\n"

        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        return chunks

    def _determine_modality(self, artifact: ProcessedArtifact) -> str:
        """Determine the primary modality of processed artifact.

        Args:
            artifact: Processed artifact

        Returns:
            Primary modality string
        """
        modalities = []
        if artifact.text_chunks:
            modalities.append(ModalityType.TEXT.value)
        if artifact.table_data:
            modalities.append(ModalityType.TABLE.value)
        if artifact.ocr_results:
            modalities.append(ModalityType.IMAGE.value)

        if len(modalities) > 1:
            return ModalityType.COMPOUND.value
        return modalities[0] if modalities else "unknown"

    def extract_chunks_for_ingestion(
        self,
        artifact: ArtifactBundle,
    ) -> List[Dict[str, Any]]:
        """Extract chunks ready for ingestion into retrieval engine.

        Args:
            artifact: Artifact bundle to process

        Returns:
            List of chunk dictionaries suitable for ingestion
        """
        processed = self.process_artifact_bundle(artifact)
        chunks = []

        # Add text chunks
        for idx, text in enumerate(processed.text_chunks):
            chunks.append({
                "chunk_id": f"{artifact.source_id}_text_{idx}",
                "source_id": artifact.source_id,
                "source_type": artifact.source_type.value,
                "modality": "text",
                "text": text,
                "token_count": len(text.split()),
            })

        # Add table chunks
        for table in processed.table_data:
            table_text = self._table_to_text(table)
            chunks.append({
                "chunk_id": table.table_id,
                "source_id": artifact.source_id,
                "source_type": artifact.source_type.value,
                "modality": "table",
                "text": table_text,
                "token_count": len(table_text.split()),
                "metadata": {"table_headers": table.headers},
            })

        # Add OCR chunks
        for ocr in processed.ocr_results:
            chunks.append({
                "chunk_id": ocr.image_id,
                "source_id": artifact.source_id,
                "source_type": artifact.source_type.value,
                "modality": "image",
                "text": ocr.text,
                "token_count": len(ocr.text.split()),
                "metadata": {"ocr_confidence": ocr.ocr_confidence},
            })

        return chunks

    def _table_to_text(self, table: ExtractedTable) -> str:
        """Convert table to searchable text.

        Args:
            table: Extracted table

        Returns:
            Text representation of table
        """
        lines = []

        if table.caption:
            lines.append(f"Table: {table.caption}")

        if table.headers:
            lines.append(" | ".join(table.headers))
            lines.append(" | ".join(["---"] * len(table.headers)))

        for row in table.rows:
            lines.append(" | ".join(row))

        return "\n".join(lines)


# Import ArtifactBundle for type hints
from .retrieval import ArtifactBundle
