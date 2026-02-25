import hashlib
import json
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Iterable, List, Sequence


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

    def query(self, query_text: str, top_k: int = 5) -> List[RankedChunk]:
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

        top = scored[: max(1, top_k)]

        if not top:
            return []

        max_score = top[0].score
        for rank, item in enumerate(top):
            normalized = item.score / max(max_score, 1e-9)
            coverage = len(item.matched_terms) / max(len(plan.tokens), 1)
            extraction = item.score_breakdown.get("extraction", 1.0)
            decay = 1.0 - (rank * 0.12)
            confidence = (normalized * 0.55) + (coverage * 0.25) + (extraction * 0.20)
            item.confidence = round(max(0.0, min(1.0, confidence * decay)), 4)

        return top


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


def _hash_id(source_id: str, unit_idx: int, chunk_idx: int, text: str) -> str:
    h = hashlib.sha1()
    h.update(source_id.encode("utf-8"))
    h.update(str(unit_idx).encode("utf-8"))
    h.update(str(chunk_idx).encode("utf-8"))
    h.update(text.encode("utf-8"))
    return h.hexdigest()[:16]
