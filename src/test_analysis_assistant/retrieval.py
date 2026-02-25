import hashlib
import json
import math
from pathlib import Path
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Iterable, List, Optional, Sequence


_WORD_RE = re.compile(r"[A-Za-z0-9_]+")


class SourceType(str, Enum):
    CODE_SNIPPET = "code_snippet"
    REPOSITORY = "repository"
    REQUIREMENTS = "requirements"
    SYSTEM_ANALYSIS = "system_analysis"
    KNOWLEDGE = "knowledge"


@dataclass
class IngestDocument:
    source_id: str
    source_type: SourceType
    content: Any
    modality: str = "text"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Chunk:
    chunk_id: str
    source_id: str
    source_type: SourceType
    modality: str
    text: str
    token_count: int
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RankedChunk:
    chunk: Chunk
    score: float
    confidence: float
    matched_terms: List[str] = field(default_factory=list)
    score_breakdown: Dict[str, float] = field(default_factory=dict)


@dataclass
class QueryPlan:
    query_text: str
    tokens: List[str]
    intent_labels: List[str] = field(default_factory=list)
    preferred_source_types: List[SourceType] = field(default_factory=list)
    preferred_modalities: List[str] = field(default_factory=list)


@dataclass
class ExtractedUnit:
    text: str
    modality: str
    extraction_confidence: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class RetrievalEngine:
    def __init__(self, chunk_size: int = 360, chunk_overlap: int = 40) -> None:
        self._chunk_size = max(80, chunk_size)
        self._chunk_overlap = max(0, min(chunk_overlap, self._chunk_size // 2))
        self._chunks: List[Chunk] = []

    def ingest_documents(self, docs: Sequence[IngestDocument]) -> List[Chunk]:
        chunked: List[Chunk] = []
        for doc in docs:
            text_units = _extract_text_units(doc)
            for unit_idx, unit in enumerate(text_units):
                for chunk_idx, piece in enumerate(_chunk_text(unit.text, self._chunk_size, self._chunk_overlap)):
                    if not piece.strip():
                        continue
                    chunk_metadata = dict(doc.metadata)
                    chunk_metadata.update(
                        {
                            "unit_index": unit_idx,
                            "extraction_confidence": round(unit.extraction_confidence, 4),
                            **unit.metadata,
                        }
                    )
                    chunk = Chunk(
                        chunk_id=_hash_id(doc.source_id, unit_idx, chunk_idx, piece),
                        source_id=doc.source_id,
                        source_type=doc.source_type,
                        modality=_effective_modality(unit.modality, unit.text),
                        text=piece,
                        token_count=len(_tokenize(piece)),
                        metadata=chunk_metadata,
                    )
                    chunked.append(chunk)

        self._chunks.extend(chunked)
        return chunked

    def build_query_plan(self, query_text: str) -> QueryPlan:
        tokens = sorted(set(_tokenize(query_text)))
        token_set = set(tokens)
        intent_rules = {
            "failure_clustering": {"cluster", "group", "pattern", "recurring", "flaky", "flake"},
            "root_cause": {"root", "cause", "traceback", "hypothesis", "why"},
            "test_gap": {"gap", "missing", "coverage", "untested", "negative"},
            "risk_prioritization": {"risk", "prioritize", "priority", "blocking", "release", "p0"},
            "actionable_plan": {"plan", "mitigation", "next", "actions", "roadmap"},
        }
        intent_labels = [name for name, words in intent_rules.items() if token_set.intersection(words)]

        source_preferences_by_intent = {
            "failure_clustering": [SourceType.REPOSITORY, SourceType.CODE_SNIPPET, SourceType.SYSTEM_ANALYSIS],
            "root_cause": [SourceType.CODE_SNIPPET, SourceType.SYSTEM_ANALYSIS, SourceType.REPOSITORY],
            "test_gap": [SourceType.REQUIREMENTS, SourceType.SYSTEM_ANALYSIS, SourceType.REPOSITORY],
            "risk_prioritization": [SourceType.REQUIREMENTS, SourceType.SYSTEM_ANALYSIS, SourceType.KNOWLEDGE],
            "actionable_plan": [SourceType.REQUIREMENTS, SourceType.SYSTEM_ANALYSIS, SourceType.CODE_SNIPPET],
        }
        preferred_source_types: List[SourceType] = []
        for intent in intent_labels:
            preferred_source_types.extend(source_preferences_by_intent.get(intent, []))
        if not preferred_source_types:
            preferred_source_types = [
                SourceType.REQUIREMENTS,
                SourceType.SYSTEM_ANALYSIS,
                SourceType.CODE_SNIPPET,
                SourceType.REPOSITORY,
                SourceType.KNOWLEDGE,
            ]

        preferred_modalities = ["text"]
        if token_set.intersection({"table", "matrix", "spreadsheet"}):
            preferred_modalities.append("table")
        if token_set.intersection({"image", "diagram", "screenshot", "ocr"}):
            preferred_modalities.append("image")
            preferred_modalities.append("image_ocr_stub")

        return QueryPlan(
            query_text=query_text,
            tokens=tokens,
            intent_labels=intent_labels,
            preferred_source_types=_dedupe(preferred_source_types),
            preferred_modalities=_dedupe(preferred_modalities),
        )

    def query(self, query_text: str, top_k: int = 5, diversify: bool = True) -> List[RankedChunk]:
        plan = self.build_query_plan(query_text)
        query_tokens = set(plan.tokens)
        if not self._chunks or not plan.tokens:
            return []

        scored: List[RankedChunk] = []
        fallback: List[RankedChunk] = []
        for chunk in self._chunks:
            chunk_tokens = set(_tokenize(chunk.text))
            overlap = sorted(query_tokens.intersection(chunk_tokens))
            lexical_score = len(overlap) / max(len(plan.tokens), 1)
            source_boost = _source_weight(chunk.source_type)
            intent_boost = _intent_alignment(plan, chunk)
            modality_boost = _modality_alignment(plan, chunk.modality)
            extraction_quality = _extraction_quality(chunk)
            score = (
                lexical_score * 0.55
                + source_boost * 0.15
                + intent_boost * 0.15
                + modality_boost * 0.05
                + extraction_quality * 0.10
            )
            breakdown = {
                "lexical": round(lexical_score, 4),
                "source": round(source_boost, 4),
                "intent": round(intent_boost, 4),
                "modality": round(modality_boost, 4),
                "extraction": round(extraction_quality, 4),
            }
            if overlap:
                scored.append(
                    RankedChunk(
                        chunk=chunk,
                        score=score,
                        confidence=0.0,
                        matched_terms=overlap,
                        score_breakdown=breakdown,
                    )
                )
            else:
                # Fallback path for sparse/noisy corpora: keep low-score candidates so
                # downstream prompting can still show best-effort evidence and gaps.
                fallback.append(
                    RankedChunk(
                        chunk=chunk,
                        score=score * 0.15,
                        confidence=0.0,
                        matched_terms=[],
                        score_breakdown=breakdown,
                    )
                )

        scored.sort(
            key=lambda item: (
                -item.score,
                -len(item.matched_terms),
                item.chunk.source_id,
                item.chunk.chunk_id,
            )
        )
        if len(scored) < max(1, top_k):
            fallback.sort(
                key=lambda item: (
                    -item.score,
                    item.chunk.source_id,
                    item.chunk.chunk_id,
                )
            )
            scored.extend(fallback)

        limit = max(1, top_k)
        top = _select_diverse_top(scored, limit) if diversify else scored[:limit]

        if not top:
            return []

        max_score = top[0].score
        source_span = len({item.chunk.source_id for item in top})
        modality_span = len({item.chunk.modality for item in top})
        diversity_bonus = ((source_span / len(top)) * 0.6) + ((modality_span / len(top)) * 0.4)
        for rank, item in enumerate(top):
            normalized = item.score / max(max_score, 1e-9)
            coverage = len(item.matched_terms) / max(len(plan.tokens), 1)
            extraction = item.score_breakdown.get("extraction", 1.0)
            decay = 1.0 - (rank * 0.12)
            confidence = (
                (normalized * 0.50)
                + (coverage * 0.22)
                + (extraction * 0.18)
                + (diversity_bonus * 0.10)
            )
            item.confidence = round(max(0.0, min(1.0, confidence * decay)), 4)

        return top


class MultiSourceIngestor:
    def __init__(self, engine: RetrievalEngine) -> None:
        self._engine = engine

    def ingest_raw(
        self,
        source_id: str,
        source_type: SourceType,
        content: Any,
        modality: str = "text",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Chunk]:
        doc = IngestDocument(
            source_id=source_id,
            source_type=source_type,
            content=content,
            modality=modality,
            metadata=metadata or {},
        )
        return self._engine.ingest_documents([doc])

    def ingest_repository(
        self,
        repo_root: str,
        max_files: int = 200,
        include_extensions: Optional[Sequence[str]] = None,
    ) -> List[Chunk]:
        root = Path(repo_root)
        if not root.exists() or not root.is_dir():
            raise ValueError(f"Repository path does not exist or is not a directory: {repo_root}")

        default_extensions = {
            ".py",
            ".md",
            ".rst",
            ".txt",
            ".json",
            ".yaml",
            ".yml",
            ".toml",
            ".ini",
            ".cfg",
        }
        allow = set(ext.lower() for ext in (include_extensions or list(default_extensions)))

        docs: List[IngestDocument] = []
        for path in sorted(root.rglob("*")):
            if len(docs) >= max_files:
                break
            if not path.is_file():
                continue
            if path.suffix.lower() not in allow:
                continue
            rel_path = path.relative_to(root).as_posix()
            try:
                text = path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            if not text.strip():
                continue
            docs.append(
                IngestDocument(
                    source_id=f"repo:{rel_path}",
                    source_type=SourceType.REPOSITORY,
                    content=text,
                    modality="text",
                    metadata={"path": rel_path, "extension": path.suffix.lower()},
                )
            )

        if not docs:
            return []
        return self._engine.ingest_documents(docs)

    def ingest_requirements_markdown(self, source_id: str, markdown: str) -> List[Chunk]:
        return self.ingest_raw(
            source_id=source_id,
            source_type=SourceType.REQUIREMENTS,
            content=markdown,
            modality="markdown_mixed",
            metadata={"format": "markdown"},
        )


def build_analysis_prompt(question: str, ranked_context: Sequence[RankedChunk]) -> str:
    lines = [
        "You are an AI-native test analysis assistant.",
        "Use the retrieved context to produce: failure clustering, root-cause hypotheses, test-gap analysis, risk-based prioritization, and an actionable plan.",
        f"Question: {question}",
        "",
        "Context:",
    ]
    if not ranked_context:
        lines.append("- (no retrieved context; answer with low-confidence assumptions and call out missing evidence)")
    else:
        for idx, item in enumerate(ranked_context, start=1):
            snippet = item.chunk.text.replace("\n", " ").strip()
            if len(snippet) > 220:
                snippet = snippet[:217] + "..."
            lines.append(
                f"[{idx}] source={item.chunk.source_id} type={item.chunk.source_type.value} modality={item.chunk.modality} confidence={item.confidence:.2f} terms={','.join(item.matched_terms)}"
            )
            lines.append(
                "    score_parts="
                + ",".join(
                    f"{name}:{value:.2f}" for name, value in sorted(item.score_breakdown.items())
                )
            )
            lines.append(f"    {snippet}")

    lines.extend(
        [
            "",
            "Response requirements:",
            "1) cite context ids like [1], [2] for every key claim",
            "2) provide confidence per section (high/medium/low) based on evidence coverage",
            "3) explicitly list missing artifacts that would improve confidence",
        ]
    )
    return "\n".join(lines)


def _extract_text_units(doc: IngestDocument) -> List[ExtractedUnit]:
    if doc.modality == "markdown_mixed" and isinstance(doc.content, str):
        return _extract_markdown_mixed_units(doc.content)
    if doc.modality == "table":
        return [
            ExtractedUnit(
                text=_table_to_text(doc.content),
                modality="table",
                extraction_confidence=0.85,
            )
        ]
    if doc.modality == "image":
        image_text = _image_to_ocr_stub(doc.content)
        extraction_confidence = 0.25 if image_text.startswith("[OCR_STUB]") else 0.65
        return [
            ExtractedUnit(
                text=image_text,
                modality="image",
                extraction_confidence=extraction_confidence,
            )
        ]
    if isinstance(doc.content, str):
        return [ExtractedUnit(text=doc.content, modality="text", extraction_confidence=1.0)]
    if isinstance(doc.content, dict):
        return [
            ExtractedUnit(
                text=json.dumps(doc.content, sort_keys=True),
                modality=doc.modality or "text",
                extraction_confidence=0.9,
            )
        ]
    if isinstance(doc.content, list):
        return [
            ExtractedUnit(
                text="\n".join(str(item) for item in doc.content),
                modality=doc.modality or "text",
                extraction_confidence=0.9,
            )
        ]
    return [ExtractedUnit(text=str(doc.content), modality=doc.modality or "text", extraction_confidence=0.8)]


def _table_to_text(content: Any) -> str:
    if isinstance(content, dict) and isinstance(content.get("rows"), list):
        parts: List[str] = []
        for idx, row in enumerate(content["rows"], start=1):
            if isinstance(row, dict):
                rendered = ", ".join(f"{k}={v}" for k, v in sorted(row.items()))
            else:
                rendered = str(row)
            parts.append(f"row{idx}: {rendered}")
        return "\n".join(parts) if parts else "table: empty"
    return f"table_raw: {content}"


def _image_to_ocr_stub(content: Any) -> str:
    if isinstance(content, dict):
        if isinstance(content.get("ocr_text"), str) and content["ocr_text"].strip():
            return content["ocr_text"]
        image_path = str(content.get("image_path", "unknown_image"))
        return f"[OCR_STUB] no OCR pipeline connected for {image_path}."
    return "[OCR_STUB] image payload provided without OCR text."


def _chunk_text(text: str, size: int, overlap: int) -> Iterable[str]:
    cleaned = text.strip()
    if len(cleaned) <= size:
        return [cleaned]

    chunks: List[str] = []
    start = 0
    while start < len(cleaned):
        end = min(len(cleaned), start + size)
        if end < len(cleaned):
            boundary = cleaned.rfind(" ", start, end)
            if boundary > start + 20:
                end = boundary
        piece = cleaned[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= len(cleaned):
            break
        start = max(end - overlap, start + 1)

    return chunks


def _extract_markdown_mixed_units(markdown: str) -> List[ExtractedUnit]:
    lines = markdown.splitlines()
    units: List[ExtractedUnit] = []

    # Preserve plain-text context while pulling table and image artifacts into
    # dedicated units for modality-aware retrieval.
    plain_lines: List[str] = []
    in_table = False
    table_lines: List[str] = []
    table_index = 0
    image_index = 0

    image_pattern = re.compile(r"!\[[^\]]*\]\(([^)]+)\)")
    for line in lines:
        image_matches = image_pattern.findall(line)
        if image_matches:
            for image_path in image_matches:
                image_text = _image_to_ocr_stub({"image_path": image_path})
                extraction_confidence = 0.25 if image_text.startswith("[OCR_STUB]") else 0.65
                units.append(
                    ExtractedUnit(
                        text=image_text,
                        modality="image",
                        extraction_confidence=extraction_confidence,
                        metadata={"image_index": image_index, "image_path": image_path},
                    )
                )
                image_index += 1
            continue

        looks_like_table = "|" in line and line.strip().startswith("|")
        if looks_like_table:
            in_table = True
            table_lines.append(line)
            continue

        if in_table:
            table_text = "\n".join(table_lines).strip()
            if table_text:
                units.append(
                    ExtractedUnit(
                        text=table_text,
                        modality="table",
                        extraction_confidence=0.82,
                        metadata={"table_index": table_index},
                    )
                )
                table_index += 1
            table_lines = []
            in_table = False

        plain_lines.append(line)

    if table_lines:
        table_text = "\n".join(table_lines).strip()
        if table_text:
            units.append(
                ExtractedUnit(
                    text=table_text,
                    modality="table",
                    extraction_confidence=0.82,
                    metadata={"table_index": table_index},
                )
            )

    plain_text = "\n".join(line for line in plain_lines if line.strip()).strip()
    if plain_text:
        units.append(
            ExtractedUnit(
                text=plain_text,
                modality="text",
                extraction_confidence=0.95,
                metadata={"extracted_from": "markdown_mixed"},
            )
        )

    if units:
        return units
    return [ExtractedUnit(text=markdown, modality="text", extraction_confidence=0.9)]


def _tokenize(text: str) -> List[str]:
    return [_normalize_token(token.lower()) for token in _WORD_RE.findall(text)]


def _normalize_token(token: str) -> str:
    if token.endswith("ies") and len(token) > 4:
        return token[:-3] + "y"
    if token.endswith("es") and len(token) > 4:
        return token[:-2]
    if token.endswith("s") and len(token) > 3:
        return token[:-1]
    return token


def _source_weight(source_type: SourceType) -> float:
    weights = {
        SourceType.REQUIREMENTS: 1.0,
        SourceType.SYSTEM_ANALYSIS: 0.95,
        SourceType.CODE_SNIPPET: 0.9,
        SourceType.REPOSITORY: 0.85,
        SourceType.KNOWLEDGE: 0.7,
    }
    return weights.get(source_type, 0.75)


def _effective_modality(modality: str, text: str) -> str:
    if modality == "image" and text.startswith("[OCR_STUB]"):
        return "image_ocr_stub"
    return modality


def _intent_alignment(plan: QueryPlan, chunk: Chunk) -> float:
    if chunk.source_type in plan.preferred_source_types:
        position = plan.preferred_source_types.index(chunk.source_type)
        return max(0.4, 1.0 - (position * 0.12))
    return 0.35


def _modality_alignment(plan: QueryPlan, chunk_modality: str) -> float:
    if chunk_modality in plan.preferred_modalities:
        position = plan.preferred_modalities.index(chunk_modality)
        return max(0.4, 1.0 - (position * 0.15))
    if chunk_modality == "image_ocr_stub":
        return 0.2
    return 0.5


def _extraction_quality(chunk: Chunk) -> float:
    value = chunk.metadata.get("extraction_confidence", 0.7)
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return 0.7
    return max(0.0, min(1.0, numeric))


def _dedupe(items: Sequence[Any]) -> List[Any]:
    seen = set()
    ordered: List[Any] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        ordered.append(item)
    return ordered


def _select_diverse_top(candidates: Sequence[RankedChunk], top_k: int) -> List[RankedChunk]:
    if top_k <= 0 or not candidates:
        return []
    if len(candidates) <= top_k:
        return list(candidates)

    selected: List[RankedChunk] = [candidates[0]]
    remaining = list(candidates[1:])
    lambda_score = 0.82
    novelty_penalty = 0.18

    while remaining and len(selected) < top_k:
        best_idx = 0
        best_value = float("-inf")
        for idx, candidate in enumerate(remaining):
            max_similarity = max(_chunk_similarity(candidate.chunk, item.chunk) for item in selected)
            mmr_value = (lambda_score * candidate.score) - (novelty_penalty * max_similarity)
            if mmr_value > best_value:
                best_value = mmr_value
                best_idx = idx
        selected.append(remaining.pop(best_idx))

    return selected


def _chunk_similarity(left: Chunk, right: Chunk) -> float:
    left_tokens = set(_tokenize(left.text))
    right_tokens = set(_tokenize(right.text))
    union = left_tokens.union(right_tokens)
    lexical = (len(left_tokens.intersection(right_tokens)) / len(union)) if union else 0.0
    source_penalty = 0.25 if left.source_id == right.source_id else 0.0
    modality_penalty = 0.10 if left.modality == right.modality else 0.0
    return min(1.0, lexical + source_penalty + modality_penalty)


def _hash_id(source_id: str, unit_idx: int, chunk_idx: int, text: str) -> str:
    h = hashlib.sha1()
    h.update(source_id.encode("utf-8"))
    h.update(str(unit_idx).encode("utf-8"))
    h.update(str(chunk_idx).encode("utf-8"))
    h.update(text.encode("utf-8"))
    return h.hexdigest()[:16]


class EmbeddingProvider:
    """Interface for embedding-based retrieval."""

    def encode(self, texts: List[str]) -> List[List[float]]:
        """Encode texts into embedding vectors.

        Args:
            texts: List of text strings to encode

        Returns:
            List of embedding vectors (one per input text)
        """
        raise NotImplementedError


class DummyEmbeddingProvider(EmbeddingProvider):
    """Fallback provider that uses word frequency as pseudo-embeddings."""

    def encode(self, texts: List[str]) -> List[List[float]]:
        """Encode using simplified bag-of-words approach."""
        vectors: List[List[float]] = []
        for text in texts:
            tokens = set(_tokenize(text.lower()))
            # Create a sparse vector using hash-like projection
            vector = [1.0 if t in tokens else 0.0 for t in sorted(tokens)][:64]
            # Pad to 64 dimensions
            vector.extend([0.0] * (64 - len(vector)))
            vectors.append(vector)
        return vectors


class TFIDFEmbeddingProvider(EmbeddingProvider):
    """TF-IDF based embedding provider for semantic similarity.

    This provider builds a vocabulary from all documents and computes
    TF-IDF vectors for semantic search. It provides better semantic
    matching than bag-of-words while remaining lightweight (no external deps).
    """

    def __init__(self, min_df: int = 1, max_features: int = 512) -> None:
        """Initialize the TF-IDF provider.

        Args:
            min_df: Minimum document frequency for vocabulary inclusion
            max_features: Maximum vocabulary size
        """
        self._min_df = min_df
        self._max_features = max_features
        self._vocabulary: Dict[str, int] = {}
        self._idf: Dict[str, float] = {}
        self._doc_count: int = 0

    def encode(self, texts: List[str]) -> List[List[float]]:
        """Encode texts using TF-IDF vectors.

        Args:
            texts: List of text strings to encode

        Returns:
            List of TF-IDF vectors
        """
        if not texts:
            return []

        # Build vocabulary and compute IDF
        self._build_vocabulary(texts)

        # Encode each text
        vectors: List[List[float]] = []
        for text in texts:
            vectors.append(self._text_to_vector(text))

        return vectors

    def _build_vocabulary(self, texts: List[str]) -> None:
        """Build vocabulary from texts and compute IDF scores."""
        self._doc_count = len(texts)

        # Count document frequencies
        doc_freq: Dict[str, int] = {}
        for text in texts:
            tokens = set(_tokenize(text.lower()))
            for token in tokens:
                doc_freq[token] = doc_freq.get(token, 0) + 1

        # Build vocabulary with max_features limit
        sorted_terms = sorted(
            doc_freq.items(),
            key=lambda x: x[1],
            reverse=True
        )

        self._vocabulary = {
            term: idx for idx, (term, _) in enumerate(sorted_terms)
            if doc_freq[term] >= self._min_df
        }
        self._vocabulary = dict(list(self._vocabulary.items())[:self._max_features])

        # Compute IDF scores
        self._idf = {}
        for term, df in doc_freq.items():
            if term in self._vocabulary:
                # Smooth IDF to avoid division by zero
                self._idf[term] = math.log((self._doc_count + 1) / (df + 1)) + 1

    def _text_to_vector(self, text: str) -> List[float]:
        """Convert text to TF-IDF vector."""
        tokens = _tokenize(text.lower())
        term_freq: Dict[str, int] = {}

        # Count term frequencies
        for token in tokens:
            if token in self._vocabulary:
                term_freq[token] = term_freq.get(token, 0) + 1

        # Compute TF-IDF
        vector = [0.0] * len(self._vocabulary)
        for token, freq in term_freq.items():
            idx = self._vocabulary.get(token)
            if idx is not None:
                # TF = 1 + log(freq), IDF from precomputed scores
                tf = 1.0 + math.log(freq) if freq > 0 else 0.0
                idf = self._idf.get(token, 1.0)
                vector[idx] = tf * idf

        # Normalize to unit vector
        magnitude = math.sqrt(sum(v * v for v in vector))
        if magnitude > 0:
            vector = [v / magnitude for v in vector]

        return vector


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    """Compute cosine similarity between two vectors."""
    if not a or not b:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


class HybridRetrievalEngine(RetrievalEngine):
    """Enhanced retrieval engine with hybrid (lexical + semantic) search."""

    def __init__(
        self,
        chunk_size: int = 360,
        chunk_overlap: int = 40,
        embedding_provider: Optional[EmbeddingProvider] = None,
        lexical_weight: float = 0.5,
    ) -> None:
        super().__init__(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self._embedding_provider = embedding_provider or DummyEmbeddingProvider()
        self._lexical_weight = max(0.0, min(1.0, lexical_weight))
        self._semantic_weight = 1.0 - self._lexical_weight
        self._chunk_embeddings: Dict[str, List[float]] = {}

    def ingest_documents(self, docs: Sequence[IngestDocument]) -> List[Chunk]:
        """Ingest documents and compute embeddings."""
        chunks = super().ingest_documents(docs)
        self._compute_embeddings()
        return chunks

    def _compute_embeddings(self) -> None:
        """Compute embeddings for all chunks."""
        if not self._chunks:
            return
        texts = [chunk.text for chunk in self._chunks]
        try:
            embeddings = self._embedding_provider.encode(texts)
            for chunk, embedding in zip(self._chunks, embeddings):
                self._chunk_embeddings[chunk.chunk_id] = embedding
        except Exception:
            # Fallback to dummy embeddings if encoding fails
            dummy = DummyEmbeddingProvider()
            embeddings = dummy.encode(texts)
            for chunk, embedding in zip(self._chunks, embeddings):
                self._chunk_embeddings[chunk.chunk_id] = embedding

    def query(
        self,
        query_text: str,
        top_k: int = 5,
        diversify: bool = True,
        use_hybrid: bool = True,
    ) -> List[RankedChunk]:
        """Query with hybrid lexical + semantic search.

        Args:
            query_text: The query string
            top_k: Number of results to return
            diversify: Whether to diversify results across sources
            use_hybrid: Whether to use hybrid search (True) or lexical only (False)
        """
        plan = self.build_query_plan(query_text)
        if not self._chunks or not plan.tokens:
            return []

        # Get lexical scores
        lexical_scores: Dict[str, float] = {}
        for chunk in self._chunks:
            chunk_tokens = set(_tokenize(chunk.text))
            overlap = plan.tokens  # Already normalized tokens
            lexical_score = len(set(overlap).intersection(chunk_tokens)) / max(len(plan.tokens), 1)
            lexical_scores[chunk.chunk_id] = lexical_score

        # Get semantic scores if hybrid mode
        semantic_scores: Dict[str, float] = {}
        if use_hybrid and self._chunk_embeddings:
            try:
                query_embedding = self._embedding_provider.encode([query_text])[0]
                for chunk in self._chunks:
                    chunk_emb = self._chunk_embeddings.get(chunk.chunk_id)
                    if chunk_emb:
                        sim = _cosine_similarity(query_embedding, chunk_emb)
                        semantic_scores[chunk.chunk_id] = sim
            except Exception:
                pass  # Fall back to lexical-only

        # Combine scores
        scored: List[RankedChunk] = []
        fallback: List[RankedChunk] = []

        for chunk in self._chunks:
            lex_score = lexical_scores.get(chunk.chunk_id, 0.0)
            sem_score = semantic_scores.get(chunk.chunk_id, 0.0)

            if use_hybrid and semantic_scores:
                # Hybrid combination
                final_score = (self._lexical_weight * lex_score) + (self._semantic_weight * sem_score)
                source_boost = _source_weight(chunk.source_type)
                intent_boost = _intent_alignment(plan, chunk)
                modality_boost = _modality_alignment(plan, chunk.modality)
                extraction_quality = _extraction_quality(chunk)
                score = (
                    final_score * 0.50
                    + source_boost * 0.15
                    + intent_boost * 0.15
                    + modality_boost * 0.05
                    + extraction_quality * 0.10
                )
            else:
                # Lexical-only (original behavior)
                source_boost = _source_weight(chunk.source_type)
                intent_boost = _intent_alignment(plan, chunk)
                modality_boost = _modality_alignment(plan, chunk.modality)
                extraction_quality = _extraction_quality(chunk)
                score = (
                    lex_score * 0.55
                    + source_boost * 0.15
                    + intent_boost * 0.15
                    + modality_boost * 0.05
                    + extraction_quality * 0.10
                )

            breakdown = {
                "lexical": round(lex_score, 4),
                "semantic": round(sem_score, 4) if use_hybrid else 0.0,
                "source": round(source_boost, 4),
                "intent": round(intent_boost, 4),
                "modality": round(modality_boost, 4),
                "extraction": round(extraction_quality, 4),
            }

            overlap = sorted(set(plan.tokens).intersection(set(_tokenize(chunk.text))))
            if overlap:
                scored.append(
                    RankedChunk(
                        chunk=chunk,
                        score=score,
                        confidence=0.0,
                        matched_terms=overlap,
                        score_breakdown=breakdown,
                    )
                )
            else:
                fallback.append(
                    RankedChunk(
                        chunk=chunk,
                        score=score * 0.15,
                        confidence=0.0,
                        matched_terms=[],
                        score_breakdown=breakdown,
                    )
                )

        # Sort and select results
        scored.sort(
            key=lambda item: (
                -item.score,
                -len(item.matched_terms),
                item.chunk.source_id,
                item.chunk.chunk_id,
            )
        )

        if len(scored) < max(1, top_k):
            fallback.sort(key=lambda item: (-item.score, item.chunk.source_id, item.chunk.chunk_id))
            scored.extend(fallback)

        limit = max(1, top_k)
        top = _select_diverse_top(scored, limit) if diversify else scored[:limit]

        if not top:
            return []

        # Compute confidence
        max_score = top[0].score
        source_span = len({item.chunk.source_id for item in top})
        modality_span = len({item.chunk.modality for item in top})
        diversity_bonus = ((source_span / len(top)) * 0.6) + ((modality_span / len(top)) * 0.4)

        for rank, item in enumerate(top):
            normalized = item.score / max(max_score, 1e-9)
            coverage = len(item.matched_terms) / max(len(plan.tokens), 1)
            extraction = item.score_breakdown.get("extraction", 1.0)
            decay = 1.0 - (rank * 0.12)
            confidence = (
                (normalized * 0.50)
                + (coverage * 0.22)
                + (extraction * 0.18)
                + (diversity_bonus * 0.10)
            )
            item.confidence = round(max(0.0, min(1.0, confidence * decay)), 4)

        return top


def create_hybrid_engine(
    chunk_size: int = 360,
    chunk_overlap: int = 40,
    embedding_provider: Optional[EmbeddingProvider] = None,
    lexical_weight: float = 0.5,
) -> HybridRetrievalEngine:
    """Factory function to create a hybrid retrieval engine.

    Args:
        chunk_size: Maximum tokens per chunk
        chunk_overlap: Token overlap between chunks
        embedding_provider: Optional custom embedding provider
        lexical_weight: Weight for lexical search (0-1), semantic gets 1-lexical_weight

    Returns:
        Configured HybridRetrievalEngine instance
    """
    return HybridRetrievalEngine(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        embedding_provider=embedding_provider,
        lexical_weight=lexical_weight,
    )
