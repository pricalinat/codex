import hashlib
import json
import math
from pathlib import Path
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .code_chunker import CodeAwareChunker, CodeChunk, CodeLanguage


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
class RetrievalEvidence:
    query_text: str
    query_plan: QueryPlan
    ranked_chunks: List[RankedChunk] = field(default_factory=list)
    source_bundles: List["SourceEvidenceBundle"] = field(default_factory=list)
    covered_source_types: List[SourceType] = field(default_factory=list)
    covered_modalities: List[str] = field(default_factory=list)
    missing_source_types: List[SourceType] = field(default_factory=list)
    missing_modalities: List[str] = field(default_factory=list)
    aggregate_confidence: float = 0.0
    confidence_band: str = "low"


@dataclass
class SourceEvidenceBundle:
    source_id: str
    source_type: SourceType
    modalities: List[str] = field(default_factory=list)
    chunk_ids: List[str] = field(default_factory=list)
    matched_terms: List[str] = field(default_factory=list)
    aggregate_score: float = 0.0
    aggregate_confidence: float = 0.0
    coverage_ratio: float = 0.0


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

    def query_with_expansion(
        self,
        query_text: str,
        top_k: int = 5,
        diversify: bool = True,
        max_variants: int = 4,
    ) -> List[RankedChunk]:
        """Run intent-aware query expansion with reciprocal-rank fusion.

        This is a degraded but usable fallback for richer retrieval when
        embedding-heavy expansion is unavailable.
        """
        variants = self._expand_query_variants(query_text, max_variants=max_variants)
        if len(variants) == 1:
            return self.query(query_text, top_k=top_k, diversify=diversify)

        aggregate: Dict[str, Dict[str, Any]] = {}
        per_variant_limit = max(top_k, 3)

        for variant_query, weight in variants:
            ranked = self.query(variant_query, top_k=per_variant_limit, diversify=False)
            for rank, item in enumerate(ranked):
                key = item.chunk.chunk_id
                entry = aggregate.setdefault(
                    key,
                    {
                        "chunk": item.chunk,
                        "best_score": item.score,
                        "best_confidence": item.confidence,
                        "rrf": 0.0,
                        "hits": 0,
                        "terms": set(),
                        "score_breakdown": dict(item.score_breakdown),
                    },
                )
                entry["best_score"] = max(entry["best_score"], item.score)
                entry["best_confidence"] = max(entry["best_confidence"], item.confidence)
                entry["hits"] += 1
                entry["rrf"] += weight * (1.0 / (50.0 + rank + 1.0))
                entry["terms"].update(item.matched_terms)

        if not aggregate:
            return []

        fused: List[RankedChunk] = []
        max_rrf = max(entry["rrf"] for entry in aggregate.values()) or 1.0
        for key, entry in aggregate.items():
            rrf_norm = entry["rrf"] / max_rrf
            score = (entry["best_score"] * 0.65) + (rrf_norm * 0.35)
            breakdown = dict(entry["score_breakdown"])
            breakdown["fusion_rrf"] = round(rrf_norm, 4)
            breakdown["fusion_hits"] = float(entry["hits"])
            fused.append(
                RankedChunk(
                    chunk=entry["chunk"],
                    score=score,
                    confidence=entry["best_confidence"],
                    matched_terms=sorted(entry["terms"]),
                    score_breakdown=breakdown,
                )
            )

        fused.sort(
            key=lambda item: (
                -item.score,
                -len(item.matched_terms),
                item.chunk.source_id,
                item.chunk.chunk_id,
            )
        )

        selected = _select_diverse_top(fused, max(1, top_k)) if diversify else fused[: max(1, top_k)]
        if not selected:
            return []

        max_score = selected[0].score
        for rank, item in enumerate(selected):
            normalized = item.score / max(max_score, 1e-9)
            decayed = max(0.0, 1.0 - (rank * 0.08))
            blended = (item.confidence * 0.62) + (normalized * 0.38)
            item.confidence = round(max(0.0, min(1.0, blended * decayed)), 4)

        return selected

    def retrieve_evidence(
        self,
        query_text: str,
        top_k: int = 5,
        diversify: bool = True,
        use_expansion: bool = True,
    ) -> RetrievalEvidence:
        plan = self.build_query_plan(query_text)
        ranked = (
            self.query_with_expansion(query_text, top_k=top_k, diversify=diversify)
            if use_expansion
            else self.query(query_text, top_k=top_k, diversify=diversify)
        )
        covered_sources = _dedupe([item.chunk.source_type for item in ranked])
        covered_modalities = _dedupe([item.chunk.modality for item in ranked])
        missing_sources = [stype for stype in plan.preferred_source_types if stype not in covered_sources]
        missing_modalities = [modality for modality in plan.preferred_modalities if modality not in covered_modalities]

        aggregate_confidence = _aggregate_confidence(ranked)
        source_bundles = _build_source_bundles(ranked, plan)
        if aggregate_confidence >= 0.72:
            band = "high"
        elif aggregate_confidence >= 0.45:
            band = "medium"
        else:
            band = "low"

        return RetrievalEvidence(
            query_text=query_text,
            query_plan=plan,
            ranked_chunks=ranked,
            source_bundles=source_bundles,
            covered_source_types=covered_sources,
            covered_modalities=covered_modalities,
            missing_source_types=missing_sources,
            missing_modalities=missing_modalities,
            aggregate_confidence=aggregate_confidence,
            confidence_band=band,
        )

    def _expand_query_variants(self, query_text: str, max_variants: int = 4) -> List[Tuple[str, float]]:
        plan = self.build_query_plan(query_text)
        variants: List[Tuple[str, float]] = [(query_text, 1.0)]

        intent_expansions = {
            "failure_clustering": "cluster recurring flaky failure patterns and shared signatures",
            "root_cause": "root cause traceback hypothesis and failure origin",
            "test_gap": "missing test coverage negative and edge case scenarios",
            "risk_prioritization": "release risk severity prioritization p0 blocking issues",
            "actionable_plan": "actionable mitigation plan next steps and owner assignments",
        }
        for intent in plan.intent_labels:
            expansion = intent_expansions.get(intent)
            if expansion:
                variants.append((f"{query_text} {expansion}", 0.82))

        # Always add one cross-goal pivot query to increase cross-source evidence
        # coverage when the corpus is sparse or vocabulary is fragmented.
        variants.append(
            (
                "failure clustering root cause test gap risk prioritization actionable plan",
                0.45,
            )
        )

        if "table" in plan.preferred_modalities:
            variants.append((f"{query_text} table matrix evidence", 0.60))
        if "image" in plan.preferred_modalities or "image_ocr_stub" in plan.preferred_modalities:
            variants.append((f"{query_text} image ocr diagram evidence", 0.60))

        deduped: List[Tuple[str, float]] = []
        seen = set()
        for variant in variants:
            if variant[0] in seen:
                continue
            seen.add(variant[0])
            deduped.append(variant)
            if len(deduped) >= max(1, max_variants):
                break
        return deduped


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


class CodeAwareIngestor:
    """Enhanced ingestor that uses code-structure-aware chunking.

    This ingestor respects function, class, and module boundaries when
    chunking code files, providing better context for test analysis.
    """

    def __init__(
        self,
        engine: RetrievalEngine,
        chunk_size: int = 360,
        chunk_overlap: int = 40,
    ) -> None:
        self._engine = engine
        self._code_chunker = CodeAwareChunker(
            max_chunk_tokens=chunk_size,
            overlap_tokens=chunk_overlap,
        )

    def ingest_repository(
        self,
        repo_root: str,
        max_files: int = 200,
        include_extensions: Optional[Sequence[str]] = None,
    ) -> List[Chunk]:
        """Ingest repository files with code-aware chunking.

        Args:
            repo_root: Path to repository root
            max_files: Maximum number of files to ingest
            include_extensions: File extensions to include

        Returns:
            List of Chunk objects
        """
        root = Path(repo_root)
        if not root.exists() or not root.is_dir():
            raise ValueError(f"Repository path does not exist or is not a directory: {repo_root}")

        default_extensions = {".py", ".js", ".ts", ".jsx", ".tsx", ".java", ".go", ".rs"}
        allow = set(ext.lower() for ext in (include_extensions or list(default_extensions)))

        chunks: List[Chunk] = []
        for path in sorted(root.rglob("*")):
            if len(chunks) >= max_files:
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

            # Use code-aware chunking
            code_chunks = self._code_chunker.chunk(text, rel_path)
            for cc in code_chunks:
                chunks.append(Chunk(
                    chunk_id=cc.chunk_id,
                    source_id=f"repo:{rel_path}",
                    source_type=SourceType.REPOSITORY,
                    modality="code",
                    text=cc.text,
                    token_count=cc.token_count,
                    metadata={
                        "path": rel_path,
                        "extension": path.suffix.lower(),
                        "language": cc.language.value,
                        "unit_name": cc.metadata.get("unit_name"),
                        "unit_type": cc.metadata.get("unit_type"),
                        "chunk_type": cc.chunk_type,
                        "start_line": cc.start_line,
                        "end_line": cc.end_line,
                        "extraction_confidence": 0.95,
                    },
                ))

        if chunks:
            self._engine._chunks.extend(chunks)
        return chunks

    def ingest_code_snippet(
        self,
        source_id: str,
        code_content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Chunk]:
        """Ingest a code snippet with structure-aware chunking.

        Args:
            source_id: Source identifier
            code_content: Code content
            metadata: Optional metadata

        Returns:
            List of Chunk objects
        """
        code_chunks = self._code_chunker.chunk(code_content, source_id)
        chunks: List[Chunk] = []

        for cc in code_chunks:
            chunks.append(Chunk(
                chunk_id=cc.chunk_id,
                source_id=source_id,
                source_type=SourceType.CODE_SNIPPET,
                modality="code",
                text=cc.text,
                token_count=cc.token_count,
                metadata={
                    "language": cc.language.value,
                    "unit_name": cc.metadata.get("unit_name"),
                    "unit_type": cc.metadata.get("unit_type"),
                    "chunk_type": cc.chunk_type,
                    "start_line": cc.start_line,
                    "end_line": cc.end_line,
                    "extraction_confidence": 0.95,
                    **(metadata or {}),
                },
            ))

        if chunks:
            self._engine._chunks.extend(chunks)
        return chunks

    def ingest_requirements_markdown(self, source_id: str, markdown: str) -> List[Chunk]:
        """Ingest requirements markdown document.

        Args:
            source_id: Source identifier
            markdown: Markdown content

        Returns:
            List of Chunk objects
        """
        # For markdown/requirements, use basic text chunking via the engine
        # but add metadata indicating it's a requirement
        return self._engine.ingest_documents([
            IngestDocument(
                source_id=source_id,
                source_type=SourceType.REQUIREMENTS,
                content=markdown,
                modality="markdown_mixed",
                metadata={"format": "markdown"},
            )
        ])

    def ingest_raw(
        self,
        source_id: str,
        source_type: SourceType,
        content: Any,
        modality: str = "text",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Chunk]:
        """Ingest raw content.

        Args:
            source_id: Source identifier
            source_type: Type of source
            content: Content to ingest
            modality: Content modality
            metadata: Optional metadata

        Returns:
            List of Chunk objects
        """
        return self._engine.ingest_documents([
            IngestDocument(
                source_id=source_id,
                source_type=source_type,
                content=content,
                modality=modality,
                metadata=metadata or {},
            )
        ])


def create_code_aware_engine(
    chunk_size: int = 360,
    chunk_overlap: int = 40,
) -> tuple:
    """Factory to create a retrieval engine with code-aware ingestion.

    Args:
        chunk_size: Maximum tokens per chunk
        chunk_overlap: Token overlap between chunks

    Returns:
        Tuple of (RetrievalEngine, CodeAwareIngestor)
    """
    engine = RetrievalEngine(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    ingestor = CodeAwareIngestor(engine, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return engine, ingestor


def build_analysis_prompt(
    question: str,
    ranked_context: Sequence[RankedChunk],
    source_bundles: Optional[Sequence[SourceEvidenceBundle]] = None,
) -> str:
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
    if source_bundles:
        lines.extend(["", "Source bundle summary:"])
        for idx, bundle in enumerate(source_bundles, start=1):
            lines.append(
                f"[S{idx}] source={bundle.source_id} type={bundle.source_type.value} "
                f"modalities={','.join(bundle.modalities)} "
                f"confidence={bundle.aggregate_confidence:.2f} "
                f"coverage={bundle.coverage_ratio:.2f} "
                f"terms={','.join(bundle.matched_terms)}"
            )

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
    if doc.modality == "compound":
        return _extract_compound_units(doc.content)
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


def _extract_compound_units(content: Any) -> List[ExtractedUnit]:
    if not isinstance(content, dict):
        return [
            ExtractedUnit(
                text=str(content),
                modality="text",
                extraction_confidence=0.75,
                metadata={"unit_kind": "fallback"},
            )
        ]

    units: List[ExtractedUnit] = []
    text_parts: List[str] = []

    for key in ("text", "summary", "body"):
        value = content.get(key)
        if isinstance(value, str) and value.strip():
            text_parts.append(value.strip())

    if text_parts:
        units.append(
            ExtractedUnit(
                text="\n\n".join(text_parts),
                modality="text",
                extraction_confidence=0.94,
                metadata={"unit_kind": "text"},
            )
        )

    raw_tables: List[Any] = []
    if isinstance(content.get("tables"), list):
        raw_tables.extend(content["tables"])
    elif content.get("tables") is not None:
        raw_tables.append(content["tables"])
    if content.get("table") is not None:
        raw_tables.append(content["table"])

    for table_idx, table_payload in enumerate(raw_tables):
        units.append(
            ExtractedUnit(
                text=_table_to_text(table_payload),
                modality="table",
                extraction_confidence=0.84,
                metadata={"unit_kind": "table", "table_index": table_idx},
            )
        )

    raw_images: List[Any] = []
    if isinstance(content.get("images"), list):
        raw_images.extend(content["images"])
    elif content.get("images") is not None:
        raw_images.append(content["images"])
    if content.get("image") is not None:
        raw_images.append(content["image"])

    for image_idx, image_payload in enumerate(raw_images):
        normalized_payload = image_payload
        if isinstance(image_payload, str):
            normalized_payload = {"image_path": image_payload}
        image_text = _image_to_ocr_stub(normalized_payload)
        extraction_confidence = 0.30 if image_text.startswith("[OCR_STUB]") else 0.68
        units.append(
            ExtractedUnit(
                text=image_text,
                modality="image",
                extraction_confidence=extraction_confidence,
                metadata={"unit_kind": "image", "image_index": image_idx},
            )
        )

    if units:
        return units

    return [
        ExtractedUnit(
            text=json.dumps(content, sort_keys=True),
            modality="text",
            extraction_confidence=0.78,
            metadata={"unit_kind": "structured_fallback"},
        )
    ]


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
    quality = max(0.0, min(1.0, numeric))

    # Provenance metadata from multimodal/compound extraction increases trust.
    if chunk.metadata.get("unit_kind"):
        quality += 0.03
    if chunk.metadata.get("origin_path") or chunk.metadata.get("image_path"):
        quality += 0.02
    if chunk.modality == "image_ocr_stub":
        quality -= 0.10

    return max(0.0, min(1.0, quality))


def _position_score(chunk: Chunk, total_chunks: int = 100) -> float:
    """Score based on chunk position in document.

    Earlier chunks tend to contain more important information
    like introductions, key concepts, and setup.

    Args:
        chunk: The chunk to score
        total_chunks: Approximate total chunks in the corpus

    Returns:
        Position score between 0 and 1
    """
    # Extract position from chunk_id or metadata
    start_line = chunk.metadata.get("start_line", 1)
    chunk_index = chunk.metadata.get("unit_index", 0)

    # Earlier chunks get higher scores
    # Use a decay factor based on position
    position_factor = 1.0 / (1.0 + 0.01 * chunk_index)

    # Also consider start_line - earlier lines get slight boost
    line_factor = 1.0 / (1.0 + 0.001 * max(0, start_line - 1))

    return 0.3 * position_factor + 0.2 * line_factor + 0.5


def _source_authority(chunk: Chunk) -> float:
    """Score based on source authority and reliability.

    Different source types have different reliability levels
    for test analysis purposes.

    Args:
        chunk: The chunk to score

    Returns:
        Authority score between 0 and 1
    """
    # Base authority by source type
    authority_by_type = {
        SourceType.REQUIREMENTS: 1.0,  # Requirements are authoritative
        SourceType.SYSTEM_ANALYSIS: 0.95,  # System docs are authoritative
        SourceType.CODE_SNIPPET: 0.9,  # Code is factual
        SourceType.REPOSITORY: 0.85,  # Repo code is factual
        SourceType.KNOWLEDGE: 0.7,  # Knowledge docs may vary
    }
    base_authority = authority_by_type.get(chunk.source_type, 0.75)

    # Adjust by modality - code modality is more precise
    modality_bonus = 0.0
    if chunk.modality == "code":
        modality_bonus = 0.1

    # Adjust by unit type if available (for code-aware chunks)
    unit_type = chunk.metadata.get("unit_type")
    if unit_type == "class":
        modality_bonus += 0.05  # Classes are often key abstractions
    elif unit_type == "function":
        modality_bonus += 0.03  # Functions are important

    return min(1.0, base_authority + modality_bonus)


def _chunk_completeness(chunk: Chunk, avg_chunk_size: int = 360) -> float:
    """Score based on chunk completeness.

    Chunks that are well-formed and complete (not truncated)
    are more reliable.

    Args:
        chunk: The chunk to score
        avg_chunk_size: Expected average chunk size

    Returns:
        Completeness score between 0 and 1
    """
    token_count = chunk.token_count
    if token_count == 0:
        return 0.1

    # Ideal chunks are close to average size
    size_ratio = token_count / avg_chunk_size
    if 0.5 <= size_ratio <= 1.2:
        completeness = 1.0
    elif size_ratio < 0.5:
        # Very small chunks might be incomplete
        completeness = 0.5 + size_ratio
    else:
        # Larger chunks might be truncated
        completeness = max(0.7, 1.0 - 0.1 * (size_ratio - 1.2))

    # Check for truncation indicators
    text = chunk.text
    if text.endswith("...") or text.endswith("[truncated]"):
        completeness *= 0.8

    return max(0.1, min(1.0, completeness))


def compute_enhanced_confidence(
    chunk: Chunk,
    query_tokens: List[str],
    matched_terms: List[str],
    score_breakdown: Dict[str, float],
    rank: int,
    total_results: int,
) -> float:
    """Compute enhanced confidence score with multiple factors.

    This function combines multiple signals to produce a more accurate
    confidence score for retrieval results.

    Args:
        chunk: The retrieved chunk
        query_tokens: Normalized query tokens
        matched_terms: Terms that matched in this chunk
        score_breakdown: Breakdown of retrieval scores
        rank: Position in results (0-indexed)
        total_results: Total number of results

    Returns:
        Enhanced confidence score between 0 and 1
    """
    if not chunk or not matched_terms:
        return 0.0

    # 1. Term coverage (how many query terms are matched)
    coverage = len(matched_terms) / max(len(query_tokens), 1)

    # 2. Extraction quality from chunk metadata
    extraction = _extraction_quality(chunk)

    # 3. Position score
    position = _position_score(chunk)

    # 4. Source authority
    authority = _source_authority(chunk)

    # 5. Chunk completeness
    completeness = _chunk_completeness(chunk)

    # 6. Rank-based decay (lower ranks get slight penalty)
    rank_decay = 1.0 - (rank * 0.08)

    # 7. Score-based confidence from retrieval
    base_score = score_breakdown.get("lexical", 0.0) + score_breakdown.get("semantic", 0.0)
    score_confidence = min(1.0, base_score * 2)

    # Weighted combination
    confidence = (
        coverage * 0.20 +
        extraction * 0.18 +
        position * 0.08 +
        authority * 0.15 +
        completeness * 0.12 +
        score_confidence * 0.17 +
        rank_decay * 0.10
    )

    return max(0.0, min(1.0, confidence))


def _aggregate_confidence(ranked: Sequence[RankedChunk]) -> float:
    if not ranked:
        return 0.0
    weighted_total = 0.0
    weight_sum = 0.0
    for idx, item in enumerate(ranked):
        weight = 1.0 / (idx + 1.0)
        weighted_total += item.confidence * weight
        weight_sum += weight
    if weight_sum == 0:
        return 0.0
    return round(max(0.0, min(1.0, weighted_total / weight_sum)), 4)


def _build_source_bundles(
    ranked: Sequence[RankedChunk],
    plan: QueryPlan,
    max_bundles: int = 6,
) -> List[SourceEvidenceBundle]:
    if not ranked:
        return []

    grouped: Dict[str, Dict[str, Any]] = {}
    query_tokens = set(plan.tokens)
    for item in ranked:
        key = item.chunk.source_id
        entry = grouped.setdefault(
            key,
            {
                "source_type": item.chunk.source_type,
                "modalities": set(),
                "chunk_ids": [],
                "matched_terms": set(),
                "score_weighted": 0.0,
                "confidence_weighted": 0.0,
                "weight_sum": 0.0,
            },
        )
        entry["modalities"].add(item.chunk.modality)
        entry["matched_terms"].update(item.matched_terms)
        entry["chunk_ids"].append(item.chunk.chunk_id)

        # Earlier ranked chunks contribute more strongly to source confidence.
        weight = 1.0 / (1.0 + len(entry["chunk_ids"]) * 0.6)
        entry["score_weighted"] += item.score * weight
        entry["confidence_weighted"] += item.confidence * weight
        entry["weight_sum"] += weight

    bundles: List[SourceEvidenceBundle] = []
    for source_id, entry in grouped.items():
        weight_sum = entry["weight_sum"] if entry["weight_sum"] > 0 else 1.0
        matched_terms = sorted(entry["matched_terms"])
        coverage_ratio = len(set(matched_terms).intersection(query_tokens)) / max(len(query_tokens), 1)
        bundles.append(
            SourceEvidenceBundle(
                source_id=source_id,
                source_type=entry["source_type"],
                modalities=sorted(entry["modalities"]),
                chunk_ids=entry["chunk_ids"][:4],
                matched_terms=matched_terms,
                aggregate_score=round(entry["score_weighted"] / weight_sum, 4),
                aggregate_confidence=round(entry["confidence_weighted"] / weight_sum, 4),
                coverage_ratio=round(coverage_ratio, 4),
            )
        )

    bundles.sort(
        key=lambda b: (
            -b.aggregate_confidence,
            -b.coverage_ratio,
            -len(b.modalities),
            b.source_id,
        )
    )
    return bundles[: max(1, max_bundles)]


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
