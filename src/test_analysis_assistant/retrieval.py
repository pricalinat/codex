import csv
import hashlib
import json
import math
from pathlib import Path
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .code_chunker import CodeAwareChunker, CodeChunk, CodeLanguage, detect_language, extract_code_units


_WORD_RE = re.compile(r"[A-Za-z0-9_]+")
_CONFLICT_POLARITY_PAIRS: Tuple[Tuple[str, str], ...] = (
    ("enabled", "disabled"),
    ("stable", "unstable"),
    ("pass", "fail"),
    ("success", "failure"),
    ("present", "missing"),
    ("allow", "deny"),
    ("fixed", "broken"),
    ("increase", "decrease"),
    ("high", "low"),
)
_CONFLICT_TERMS = {term for pair in _CONFLICT_POLARITY_PAIRS for term in pair}
_CONFLICT_CONTEXT_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "to",
    "was",
    "were",
    "with",
}


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
class CorpusCoverageProfile:
    source_type_counts: Dict[SourceType, int] = field(default_factory=dict)
    modality_counts: Dict[str, int] = field(default_factory=dict)
    source_type_modalities: Dict[SourceType, List[str]] = field(default_factory=dict)
    source_type_avg_extraction: Dict[SourceType, float] = field(default_factory=dict)


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
    corpus_available_source_types: List[SourceType] = field(default_factory=list)
    corpus_available_modalities: List[str] = field(default_factory=list)
    unavailable_preferred_source_types: List[SourceType] = field(default_factory=list)
    unavailable_preferred_modalities: List[str] = field(default_factory=list)
    aggregate_confidence: float = 0.0
    calibrated_confidence: float = 0.0
    confidence_band: str = "low"
    confidence_factors: Dict[str, float] = field(default_factory=dict)
    retrieval_strategy: str = "baseline"
    recovery_applied: bool = False
    recovery_queries: List[str] = field(default_factory=list)
    corpus_profile: CorpusCoverageProfile = field(default_factory=CorpusCoverageProfile)
    recommended_ingestion_actions: List[str] = field(default_factory=list)


@dataclass
class FocusedEvidence:
    focus_area: str
    query_text: str
    evidence: RetrievalEvidence


@dataclass
class AnalysisEvidencePack:
    query_text: str
    focus_results: List[FocusedEvidence] = field(default_factory=list)
    merged_evidence: RetrievalEvidence = field(
        default_factory=lambda: RetrievalEvidence(query_text="", query_plan=QueryPlan(query_text="", tokens=[]))
    )
    focus_confidence: Dict[str, float] = field(default_factory=dict)
    overall_confidence: float = 0.0


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
    reliability_score: float = 0.0


@dataclass
class ExtractedUnit:
    text: str
    modality: str
    extraction_confidence: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ArtifactBundle:
    """Normalized mixed-modality payload produced by extraction/OCR stubs."""

    source_id: str
    source_type: SourceType
    text: Optional[str] = None
    tables: List[Any] = field(default_factory=list)
    images: List[Any] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class IngestionRecord:
    """Normalized ingestion contract for multi-source pipeline adapters."""

    source_id: str
    source_type: SourceType
    payload: Any
    modality: str = "auto"
    metadata: Dict[str, Any] = field(default_factory=dict)


class RetrievalEngine:
    def __init__(self, chunk_size: int = 360, chunk_overlap: int = 40) -> None:
        self._chunk_size = max(80, chunk_size)
        self._chunk_overlap = max(0, min(chunk_overlap, self._chunk_size // 2))
        self._chunks: List[Chunk] = []

    def ingest_documents(
        self,
        docs: Sequence[IngestDocument],
        generate_source_summaries: bool = False,
    ) -> List[Chunk]:
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

        if generate_source_summaries and chunked:
            chunked.extend(_build_source_summary_chunks(chunked))

        self._chunks.extend(chunked)
        return chunked

    def build_corpus_profile(self) -> CorpusCoverageProfile:
        source_type_counts: Dict[SourceType, int] = {}
        modality_counts: Dict[str, int] = {}
        source_modalities: Dict[SourceType, set] = {}
        extraction_totals: Dict[SourceType, float] = {}
        extraction_counts: Dict[SourceType, int] = {}

        for chunk in self._chunks:
            source_type_counts[chunk.source_type] = source_type_counts.get(chunk.source_type, 0) + 1
            modality_counts[chunk.modality] = modality_counts.get(chunk.modality, 0) + 1
            source_modalities.setdefault(chunk.source_type, set()).add(chunk.modality)

            extraction = float(chunk.metadata.get("extraction_confidence", 1.0))
            extraction_totals[chunk.source_type] = extraction_totals.get(chunk.source_type, 0.0) + extraction
            extraction_counts[chunk.source_type] = extraction_counts.get(chunk.source_type, 0) + 1

        source_type_avg_extraction: Dict[SourceType, float] = {}
        for source_type, count in extraction_counts.items():
            denom = max(1, count)
            source_type_avg_extraction[source_type] = round(extraction_totals.get(source_type, 0.0) / denom, 4)

        return CorpusCoverageProfile(
            source_type_counts=source_type_counts,
            modality_counts=modality_counts,
            source_type_modalities={k: sorted(v) for k, v in source_modalities.items()},
            source_type_avg_extraction=source_type_avg_extraction,
        )

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
                + extraction_quality * 0.06
                + _source_reliability(chunk) * 0.04
            )
            breakdown = {
                "lexical": round(lexical_score, 4),
                "source": round(source_boost, 4),
                "intent": round(intent_boost, 4),
                "modality": round(modality_boost, 4),
                "extraction": round(extraction_quality, 4),
                "reliability": round(_source_reliability(chunk), 4),
                "position": round(_position_score(chunk), 4),
                "authority": round(_source_authority(chunk), 4),
                "completeness": round(_chunk_completeness(chunk, avg_chunk_size=self._chunk_size), 4),
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

        _apply_corroboration_bonus(scored)
        _apply_corroboration_bonus(fallback)

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
            enhanced_confidence = compute_enhanced_confidence(
                chunk=item.chunk,
                query_tokens=plan.tokens,
                matched_terms=item.matched_terms,
                score_breakdown=item.score_breakdown,
                rank=rank,
                total_results=len(top),
            )
            normalized = item.score / max(max_score, 1e-9)
            coverage = len(item.matched_terms) / max(len(plan.tokens), 1)
            extraction = item.score_breakdown.get("extraction", 1.0)
            decay = 1.0 - (rank * 0.12)
            legacy_confidence = (
                (normalized * 0.50)
                + (coverage * 0.22)
                + (extraction * 0.18)
                + (diversity_bonus * 0.10)
            )
            blended = (enhanced_confidence * 0.72) + (legacy_confidence * decay * 0.28)
            item.confidence = round(max(0.0, min(1.0, blended)), 4)

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
        adaptive_recovery: bool = True,
    ) -> RetrievalEvidence:
        plan = self.build_query_plan(query_text)
        corpus_profile = self.build_corpus_profile()
        requested_top_k = max(1, top_k)
        candidate_top_k = requested_top_k
        if len(self._chunks) > requested_top_k:
            candidate_top_k = min(len(self._chunks), max((requested_top_k * 3), (requested_top_k + 4)))
        ranked = (
            self.query_with_expansion(query_text, top_k=candidate_top_k, diversify=diversify)
            if use_expansion
            else self.query(query_text, top_k=candidate_top_k, diversify=diversify)
        )
        ranked = _select_coverage_aware_top(
            ranked,
            plan=plan,
            top_k=requested_top_k,
            diversify=diversify,
        )
        corpus_available_sources = _dedupe(list(corpus_profile.source_type_counts.keys()))
        corpus_available_modalities = _dedupe(list(corpus_profile.modality_counts.keys()))
        covered_sources = _dedupe([item.chunk.source_type for item in ranked])
        covered_modalities = _dedupe([item.chunk.modality for item in ranked])
        missing_sources = [stype for stype in plan.preferred_source_types if stype not in covered_sources]
        missing_modalities = [modality for modality in plan.preferred_modalities if modality not in covered_modalities]
        unavailable_sources = [
            stype for stype in plan.preferred_source_types if stype not in corpus_available_sources
        ]
        unavailable_modalities = [
            modality for modality in plan.preferred_modalities if modality not in corpus_available_modalities
        ]
        aggregate_confidence = _aggregate_confidence(ranked)
        calibrated_confidence, confidence_factors = _calibrate_retrieval_confidence(
            ranked=ranked,
            plan=plan,
            aggregate_confidence=aggregate_confidence,
            missing_sources=missing_sources,
            missing_modalities=missing_modalities,
            unavailable_sources=unavailable_sources,
            unavailable_modalities=unavailable_modalities,
        )
        recovery_applied = False
        recovery_queries: List[str] = []
        retrieval_strategy = "baseline"

        if adaptive_recovery and _should_apply_recovery(
            aggregate_confidence=aggregate_confidence,
            missing_sources=missing_sources,
            missing_modalities=missing_modalities,
        ):
            recovery_queries = _build_recovery_queries(
                query_text=query_text,
                plan=plan,
                missing_sources=missing_sources,
                missing_modalities=missing_modalities,
            )
            if recovery_queries:
                supplemental: List[RankedChunk] = []
                recovery_top_k = max(3, top_k)
                for recovery_query in recovery_queries:
                    recovered = (
                        self.query_with_expansion(recovery_query, top_k=recovery_top_k, diversify=False)
                        if use_expansion
                        else self.query(recovery_query, top_k=recovery_top_k, diversify=False)
                    )
                    supplemental.extend(recovered)

                if supplemental:
                    fused = _fuse_ranked_chunks(ranked, supplemental)
                    ranked = _select_coverage_aware_top(
                        fused,
                        plan=plan,
                        top_k=requested_top_k,
                        diversify=diversify,
                    )
                    recovery_applied = True
                    retrieval_strategy = "adaptive_recovery"
                    aggregate_confidence = _aggregate_confidence(ranked)
                    covered_sources = _dedupe([item.chunk.source_type for item in ranked])
                    covered_modalities = _dedupe([item.chunk.modality for item in ranked])
                    missing_sources = [
                        stype for stype in plan.preferred_source_types if stype not in covered_sources
                    ]
                    missing_modalities = [
                        modality for modality in plan.preferred_modalities if modality not in covered_modalities
                    ]
                    calibrated_confidence, confidence_factors = _calibrate_retrieval_confidence(
                        ranked=ranked,
                        plan=plan,
                        aggregate_confidence=aggregate_confidence,
                        missing_sources=missing_sources,
                        missing_modalities=missing_modalities,
                        unavailable_sources=unavailable_sources,
                        unavailable_modalities=unavailable_modalities,
                    )

        source_bundles = _build_source_bundles(ranked, plan)
        if calibrated_confidence >= 0.72:
            band = "high"
        elif calibrated_confidence >= 0.45:
            band = "medium"
        else:
            band = "low"
        recommended_ingestion_actions = _recommend_ingestion_actions(
            plan=plan,
            corpus_profile=corpus_profile,
            missing_sources=missing_sources,
            missing_modalities=missing_modalities,
            unavailable_sources=unavailable_sources,
            unavailable_modalities=unavailable_modalities,
        )

        return RetrievalEvidence(
            query_text=query_text,
            query_plan=plan,
            ranked_chunks=ranked,
            source_bundles=source_bundles,
            covered_source_types=covered_sources,
            covered_modalities=covered_modalities,
            missing_source_types=missing_sources,
            missing_modalities=missing_modalities,
            corpus_available_source_types=corpus_available_sources,
            corpus_available_modalities=corpus_available_modalities,
            unavailable_preferred_source_types=unavailable_sources,
            unavailable_preferred_modalities=unavailable_modalities,
            aggregate_confidence=aggregate_confidence,
            calibrated_confidence=calibrated_confidence,
            confidence_band=band,
            confidence_factors=confidence_factors,
            retrieval_strategy=retrieval_strategy,
            recovery_applied=recovery_applied,
            recovery_queries=recovery_queries,
            corpus_profile=corpus_profile,
            recommended_ingestion_actions=recommended_ingestion_actions,
        )

    def retrieve_analysis_evidence_pack(
        self,
        query_text: str,
        top_k_per_focus: int = 4,
        diversify: bool = True,
        use_expansion: bool = True,
        adaptive_recovery: bool = True,
    ) -> AnalysisEvidencePack:
        focus_queries = _build_focus_queries(query_text)
        focus_results: List[FocusedEvidence] = []
        focus_confidence: Dict[str, float] = {}
        merged_ranked: List[RankedChunk] = []
        merged_recovery_applied = False
        merged_recovery_queries: List[str] = []
        merged_strategy = "baseline"

        for focus_area, focus_query in focus_queries:
            evidence = self.retrieve_evidence(
                focus_query,
                top_k=max(1, top_k_per_focus),
                diversify=diversify,
                use_expansion=use_expansion,
                adaptive_recovery=adaptive_recovery,
            )
            focus_results.append(FocusedEvidence(focus_area=focus_area, query_text=focus_query, evidence=evidence))
            focus_confidence[focus_area] = evidence.calibrated_confidence
            merged_recovery_applied = merged_recovery_applied or evidence.recovery_applied
            if evidence.recovery_queries:
                merged_recovery_queries.extend(evidence.recovery_queries)
            if evidence.retrieval_strategy == "adaptive_recovery":
                merged_strategy = "adaptive_recovery"
            if not merged_ranked:
                merged_ranked = list(evidence.ranked_chunks)
            else:
                merged_ranked = _fuse_ranked_chunks(merged_ranked, evidence.ranked_chunks)

        merged_query = f"{query_text} failure clustering root cause test gap risk prioritization actionable plan".strip()
        merged_plan = self.build_query_plan(merged_query)
        corpus_profile = self.build_corpus_profile()
        merged_ranked = _select_coverage_aware_top(
            merged_ranked,
            plan=merged_plan,
            top_k=max(top_k_per_focus * 2, 6),
            diversify=diversify,
        )
        covered_sources = _dedupe([item.chunk.source_type for item in merged_ranked])
        covered_modalities = _dedupe([item.chunk.modality for item in merged_ranked])
        missing_sources = [stype for stype in merged_plan.preferred_source_types if stype not in covered_sources]
        missing_modalities = [modality for modality in merged_plan.preferred_modalities if modality not in covered_modalities]
        corpus_available_sources = _dedupe(list(corpus_profile.source_type_counts.keys()))
        corpus_available_modalities = _dedupe(list(corpus_profile.modality_counts.keys()))
        unavailable_sources = [stype for stype in merged_plan.preferred_source_types if stype not in corpus_available_sources]
        unavailable_modalities = [
            modality for modality in merged_plan.preferred_modalities if modality not in corpus_available_modalities
        ]
        aggregate_confidence = _aggregate_confidence(merged_ranked)
        calibrated_confidence, confidence_factors = _calibrate_retrieval_confidence(
            ranked=merged_ranked,
            plan=merged_plan,
            aggregate_confidence=aggregate_confidence,
            missing_sources=missing_sources,
            missing_modalities=missing_modalities,
            unavailable_sources=unavailable_sources,
            unavailable_modalities=unavailable_modalities,
        )
        if calibrated_confidence >= 0.72:
            confidence_band = "high"
        elif calibrated_confidence >= 0.45:
            confidence_band = "medium"
        else:
            confidence_band = "low"

        merged_evidence = RetrievalEvidence(
            query_text=merged_query,
            query_plan=merged_plan,
            ranked_chunks=merged_ranked,
            source_bundles=_build_source_bundles(merged_ranked, merged_plan),
            covered_source_types=covered_sources,
            covered_modalities=covered_modalities,
            missing_source_types=missing_sources,
            missing_modalities=missing_modalities,
            corpus_available_source_types=corpus_available_sources,
            corpus_available_modalities=corpus_available_modalities,
            unavailable_preferred_source_types=unavailable_sources,
            unavailable_preferred_modalities=unavailable_modalities,
            aggregate_confidence=aggregate_confidence,
            calibrated_confidence=calibrated_confidence,
            confidence_band=confidence_band,
            confidence_factors=confidence_factors,
            retrieval_strategy=merged_strategy,
            recovery_applied=merged_recovery_applied,
            recovery_queries=_dedupe(merged_recovery_queries),
            corpus_profile=corpus_profile,
            recommended_ingestion_actions=_recommend_ingestion_actions(
                plan=merged_plan,
                corpus_profile=corpus_profile,
                missing_sources=missing_sources,
                missing_modalities=missing_modalities,
                unavailable_sources=unavailable_sources,
                unavailable_modalities=unavailable_modalities,
            ),
        )

        overall_confidence = (
            round(sum(focus_confidence.values()) / len(focus_confidence), 4)
            if focus_confidence
            else merged_evidence.calibrated_confidence
        )

        return AnalysisEvidencePack(
            query_text=query_text,
            focus_results=focus_results,
            merged_evidence=merged_evidence,
            focus_confidence=focus_confidence,
            overall_confidence=overall_confidence,
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
        generate_source_summaries: bool = False,
    ) -> List[Chunk]:
        doc = IngestDocument(
            source_id=source_id,
            source_type=source_type,
            content=content,
            modality=modality,
            metadata=metadata or {},
        )
        return self._engine.ingest_documents(
            [doc],
            generate_source_summaries=generate_source_summaries,
        )

    def ingest_records(
        self,
        records: Sequence[IngestionRecord],
        generate_source_summaries: bool = False,
    ) -> List[Chunk]:
        docs = [_normalize_ingestion_record(record) for record in records]
        if not docs:
            return []
        return self._engine.ingest_documents(
            docs,
            generate_source_summaries=generate_source_summaries,
        )

    def ingest_repository(
        self,
        repo_root: str,
        max_files: int = 200,
        include_extensions: Optional[Sequence[str]] = None,
        generate_source_summaries: bool = False,
    ) -> List[Chunk]:
        root = Path(repo_root)
        if not root.exists() or not root.is_dir():
            raise ValueError(f"Repository path does not exist or is not a directory: {repo_root}")

        default_extensions = {
            ".py",
            ".md",
            ".markdown",
            ".mdx",
            ".rst",
            ".txt",
            ".csv",
            ".tsv",
            ".json",
            ".yaml",
            ".yml",
            ".toml",
            ".ini",
            ".cfg",
        }
        allow = set(ext.lower() for ext in (include_extensions or list(default_extensions)))

        docs: List[IngestDocument] = []
        file_records: List[Dict[str, Any]] = []
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
            modality = _infer_repository_modality(path.suffix.lower())
            content: Any = text
            metadata: Dict[str, Any] = {"path": rel_path, "extension": path.suffix.lower()}
            if modality == "table":
                content = _parse_delimited_table(text, path.suffix.lower())
                metadata["format"] = "delimited_table"
            elif modality == "markdown_mixed":
                metadata["format"] = "markdown"

            docs.append(
                IngestDocument(
                    source_id=f"repo:{rel_path}",
                    source_type=SourceType.REPOSITORY,
                    content=content,
                    modality=modality,
                    metadata=metadata,
                )
            )
            file_records.append(
                {
                    "rel_path": rel_path,
                    "extension": path.suffix.lower(),
                    "modality": "code" if path.suffix.lower() in _REPO_CODE_EXTENSIONS else modality,
                    "content": text,
                }
            )

        docs.extend(_build_repository_manifest_documents(file_records))

        if not docs:
            return []
        return self._engine.ingest_documents(
            docs,
            generate_source_summaries=generate_source_summaries,
        )

    def ingest_requirements_markdown(
        self,
        source_id: str,
        markdown: str,
        generate_source_summaries: bool = False,
    ) -> List[Chunk]:
        return self.ingest_raw(
            source_id=source_id,
            source_type=SourceType.REQUIREMENTS,
            content=markdown,
            modality="markdown_mixed",
            metadata={"format": "markdown"},
            generate_source_summaries=generate_source_summaries,
        )

    def ingest_artifact_bundle(
        self,
        bundle: ArtifactBundle,
        generate_source_summaries: bool = False,
    ) -> List[Chunk]:
        """Ingest one mixed-modality artifact bundle.

        This is a stable interface for OCR/extraction pipelines to emit a
        degraded-but-usable payload without introducing extra dependencies.
        """
        content: Dict[str, Any] = {}
        if bundle.text and bundle.text.strip():
            content["text"] = bundle.text
        if bundle.tables:
            content["tables"] = list(bundle.tables)
        if bundle.images:
            content["images"] = list(bundle.images)

        if not content:
            return []

        metadata = dict(bundle.metadata)
        metadata.setdefault("format", "artifact_bundle")

        return self.ingest_raw(
            source_id=bundle.source_id,
            source_type=bundle.source_type,
            content=content,
            modality="compound",
            metadata=metadata,
            generate_source_summaries=generate_source_summaries,
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

        code_exts = {".py", ".js", ".ts", ".jsx", ".tsx", ".java", ".go", ".rs"}
        chunks: List[Chunk] = []
        file_records: List[Dict[str, Any]] = []
        files_seen = 0
        for path in sorted(root.rglob("*")):
            if files_seen >= max_files:
                break
            if not path.is_file():
                continue
            if path.suffix.lower() not in allow:
                continue
            files_seen += 1
            rel_path = path.relative_to(root).as_posix()
            try:
                text = path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            if not text.strip():
                continue

            ext = path.suffix.lower()
            file_records.append(
                {
                    "rel_path": rel_path,
                    "extension": ext,
                    "modality": "code" if ext in code_exts else _infer_repository_modality(ext),
                    "content": text,
                }
            )
            if ext not in code_exts:
                modality = _infer_repository_modality(ext)
                content: Any = text
                metadata: Dict[str, Any] = {"path": rel_path, "extension": ext}
                if modality == "table":
                    content = _parse_delimited_table(text, ext)
                    metadata["format"] = "delimited_table"
                elif modality == "markdown_mixed":
                    metadata["format"] = "markdown"
                chunks.extend(
                    self._engine.ingest_documents(
                        [
                            IngestDocument(
                                source_id=f"repo:{rel_path}",
                                source_type=SourceType.REPOSITORY,
                                content=content,
                                modality=modality,
                                metadata=metadata,
                            )
                        ]
                    )
                )
                continue

            # Use code-aware chunking for source code files.
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

        manifest_docs = _build_repository_manifest_documents(file_records)
        if manifest_docs:
            chunks.extend(self._engine.ingest_documents(manifest_docs))
        code_chunks = [chunk for chunk in chunks if chunk.modality == "code"]
        if code_chunks:
            self._engine._chunks.extend(code_chunks)
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
        return self._engine.ingest_documents(
            [
                IngestDocument(
                    source_id=source_id,
                    source_type=SourceType.REQUIREMENTS,
                    content=markdown,
                    modality="markdown_mixed",
                    metadata={"format": "markdown"},
                )
            ]
        )

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
                f"reliability={bundle.reliability_score:.2f} "
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


def build_analysis_prompt_from_evidence(question: str, evidence: RetrievalEvidence) -> str:
    """Build an analysis prompt from full retrieval evidence metadata."""
    prompt = build_analysis_prompt(
        question=question,
        ranked_context=evidence.ranked_chunks,
        source_bundles=evidence.source_bundles,
    )

    lines = [
        "",
        "Retrieval confidence:",
        (
            f"aggregate={evidence.aggregate_confidence:.2f} "
            f"calibrated={evidence.calibrated_confidence:.2f} "
            f"band={evidence.confidence_band} "
            f"strategy={evidence.retrieval_strategy}"
        ),
    ]

    if evidence.confidence_factors:
        factor_text = ", ".join(
            f"{name}={value:.2f}" for name, value in sorted(evidence.confidence_factors.items())
        )
        lines.append(f"confidence_factors: {factor_text}")

    if evidence.missing_source_types or evidence.missing_modalities:
        lines.append(
            "Missing retrieval evidence: "
            f"source_types={[item.value for item in evidence.missing_source_types]} "
            f"modalities={evidence.missing_modalities}"
        )
    if evidence.unavailable_preferred_source_types or evidence.unavailable_preferred_modalities:
        lines.append(
            "Corpus-unavailable evidence: "
            f"source_types={[item.value for item in evidence.unavailable_preferred_source_types]} "
            f"modalities={evidence.unavailable_preferred_modalities}"
        )
    if evidence.recommended_ingestion_actions:
        lines.append("Recommended ingestion actions:")
        for action in evidence.recommended_ingestion_actions:
            lines.append(f"- {action}")

    return prompt + "\n" + "\n".join(lines)


def build_analysis_prompt_from_pack(question: str, pack: AnalysisEvidencePack) -> str:
    prompt = build_analysis_prompt_from_evidence(question=question, evidence=pack.merged_evidence)
    lines = ["", "Analysis focus coverage:"]

    for item in pack.focus_results:
        evidence = item.evidence
        lines.append(
            f"- {item.focus_area}: confidence={evidence.calibrated_confidence:.2f} "
            f"band={evidence.confidence_band} "
            f"sources={len(evidence.covered_source_types)} "
            f"modalities={len(evidence.covered_modalities)}"
        )
        if evidence.missing_source_types or evidence.missing_modalities:
            lines.append(
                f"  missing source_types={[s.value for s in evidence.missing_source_types]} "
                f"modalities={evidence.missing_modalities}"
            )

    lines.append(f"overall_focus_confidence={pack.overall_confidence:.2f}")
    return prompt + "\n" + "\n".join(lines)


def _extract_text_units(doc: IngestDocument) -> List[ExtractedUnit]:
    if doc.modality == "compound":
        return _extract_compound_units(doc.content, doc.metadata)
    if doc.modality == "markdown_mixed" and isinstance(doc.content, str):
        return _extract_markdown_mixed_units(doc.content)
    if doc.modality == "table":
        extraction_confidence = _resolve_extraction_confidence(
            payload=doc.content,
            modality="table",
            default=0.85,
            metadata=doc.metadata,
        )
        return [
            ExtractedUnit(
                text=_table_to_text(doc.content),
                modality="table",
                extraction_confidence=extraction_confidence,
            )
        ]
    if doc.modality == "image":
        image_text = _image_to_ocr_stub(doc.content)
        fallback = 0.25 if image_text.startswith("[OCR_STUB]") else 0.65
        extraction_confidence = _resolve_extraction_confidence(
            payload=doc.content,
            modality="image",
            default=fallback,
            metadata=doc.metadata,
        )
        return [
            ExtractedUnit(
                text=image_text,
                modality="image",
                extraction_confidence=extraction_confidence,
            )
        ]
    if isinstance(doc.content, str):
        fallback_defaults = {
            "code": 0.96,
            "text": 1.0,
            "markdown_mixed": 0.95,
        }
        normalized_modality = (doc.modality or "text").strip().lower()
        resolved_modality = normalized_modality if normalized_modality in {"text", "code"} else "text"
        extraction_confidence = _resolve_extraction_confidence(
            payload=doc.content,
            modality=resolved_modality,
            default=fallback_defaults.get(normalized_modality, 0.92),
            metadata=doc.metadata,
        )
        return [ExtractedUnit(text=doc.content, modality=resolved_modality, extraction_confidence=extraction_confidence)]
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


def _normalize_ingestion_record(record: IngestionRecord) -> IngestDocument:
    normalized_modality = (record.modality or "auto").strip().lower()
    normalized_payload = record.payload
    metadata = dict(record.metadata)

    if normalized_modality == "auto":
        file_reference = _resolve_record_file_reference(record.payload, record.source_type)
        if file_reference is not None:
            normalized_modality, normalized_payload, reference_metadata = file_reference
            metadata.update(reference_metadata)
            metadata.setdefault("ingestion_route", "file_reference_record")
        else:
            normalized_modality, normalized_payload = _infer_record_modality_and_payload(record.payload)
            metadata.setdefault("ingestion_route", "normalized_record")
    else:
        metadata.setdefault("ingestion_route", "explicit_record")

    if normalized_modality == "compound":
        metadata.setdefault("format", "artifact_bundle")
    if normalized_modality in {"image", "image_ocr_stub"}:
        if isinstance(normalized_payload, dict) and str(normalized_payload.get("ocr_text", "")).strip():
            metadata.setdefault("ingestion_route", "pipeline_verified")
        else:
            metadata.setdefault("ingestion_route", "pipeline_ocr_stub")

    return IngestDocument(
        source_id=record.source_id,
        source_type=record.source_type,
        content=normalized_payload,
        modality=normalized_modality,
        metadata=metadata,
    )


def _infer_record_modality_and_payload(payload: Any) -> Tuple[str, Any]:
    if isinstance(payload, dict):
        if isinstance(payload.get("markdown"), str) and payload["markdown"].strip():
            return "markdown_mixed", payload["markdown"]
        if any(key in payload for key in ("text", "summary", "body", "tables", "images", "table", "image")):
            return "compound", payload
        if "rows" in payload:
            return "table", payload
        if any(key in payload for key in ("ocr_text", "image_path", "alt_text", "image_bytes")):
            return "image", payload
        return "text", json.dumps(payload, sort_keys=True)

    if isinstance(payload, list):
        if payload and all(isinstance(item, dict) and "rows" in item for item in payload):
            return "compound", {"tables": payload}
        if payload and all(isinstance(item, dict) for item in payload):
            return "table", {"rows": payload}
        return "text", "\n".join(str(item) for item in payload)

    if isinstance(payload, str):
        if _looks_like_markdown(payload):
            return "markdown_mixed", payload
        return "text", payload

    return "text", str(payload)


def _resolve_record_file_reference(
    payload: Any,
    source_type: SourceType,
) -> Optional[Tuple[str, Any, Dict[str, Any]]]:
    if not isinstance(payload, dict):
        return None

    path_value = None
    for key in ("file_path", "path", "source_path", "artifact_path"):
        candidate = payload.get(key)
        if isinstance(candidate, str) and candidate.strip():
            path_value = candidate.strip()
            break

    if not path_value:
        return None

    referenced_path = Path(path_value)
    if not referenced_path.exists() or not referenced_path.is_file():
        return None

    ext = referenced_path.suffix.lower()
    metadata: Dict[str, Any] = {
        "origin_path": referenced_path.as_posix(),
        "referenced_file_extension": ext or "[none]",
    }
    if "alt_text" in payload:
        metadata["alt_text"] = str(payload.get("alt_text", "")).strip()

    if ext in {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp", ".tif", ".tiff"}:
        image_payload: Dict[str, Any] = {"image_path": referenced_path.as_posix()}
        if isinstance(payload.get("ocr_text"), str) and payload["ocr_text"].strip():
            image_payload["ocr_text"] = payload["ocr_text"].strip()
        if isinstance(payload.get("alt_text"), str) and payload["alt_text"].strip():
            image_payload["alt_text"] = payload["alt_text"].strip()
        return "image", image_payload, metadata

    try:
        text = referenced_path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None

    if ext in {".csv", ".tsv"}:
        return "table", _parse_delimited_table(text, ext), metadata
    if ext in {".md", ".markdown", ".mdx", ".rst"}:
        return "markdown_mixed", text, metadata
    if ext in _REPO_CODE_EXTENSIONS or source_type == SourceType.CODE_SNIPPET:
        return "code", text, metadata
    if ext == ".json":
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return "text", text, metadata
        inferred_modality, inferred_payload = _infer_record_modality_and_payload(parsed)
        return inferred_modality, inferred_payload, metadata

    return "text", text, metadata


def _looks_like_markdown(text: str) -> bool:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return False
    heading = any(line.startswith("#") for line in lines[:4])
    table = any(line.startswith("|") and line.endswith("|") for line in lines)
    image = any("![" in line and "](" in line for line in lines)
    return heading or table or image


def _extract_compound_units(
    content: Any,
    document_metadata: Optional[Dict[str, Any]] = None,
) -> List[ExtractedUnit]:
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
        text_payload = {"text": "\n\n".join(text_parts)}
        units.append(
            ExtractedUnit(
                text=text_payload["text"],
                modality="text",
                extraction_confidence=_resolve_extraction_confidence(
                    payload=text_payload,
                    modality="text",
                    default=0.94,
                    metadata=document_metadata,
                ),
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
                extraction_confidence=_resolve_extraction_confidence(
                    payload=table_payload,
                    modality="table",
                    default=0.84,
                    metadata=document_metadata,
                ),
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
        image_path = None
        alt_text = ""
        if isinstance(normalized_payload, dict):
            image_path = normalized_payload.get("image_path")
            alt_text = str(normalized_payload.get("alt_text", "")).strip()
        extraction_confidence = _resolve_extraction_confidence(
            payload=normalized_payload,
            modality="image",
            default=extraction_confidence,
            metadata=document_metadata,
        )
        units.append(
            ExtractedUnit(
                text=image_text,
                modality="image",
                extraction_confidence=extraction_confidence,
                metadata={
                    "unit_kind": "image",
                    "image_index": image_idx,
                    "image_path": image_path,
                    "alt_text": alt_text,
                },
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
        alt_text = str(content.get("alt_text", "")).strip()
        if alt_text:
            return f"[OCR_STUB] no OCR pipeline connected for {image_path}. alt_text: {alt_text}"
        return f"[OCR_STUB] no OCR pipeline connected for {image_path}."
    return "[OCR_STUB] image payload provided without OCR text."


def _resolve_extraction_confidence(
    payload: Any,
    modality: str,
    default: float,
    metadata: Optional[Dict[str, Any]] = None,
) -> float:
    candidates: List[Any] = []
    modality = (modality or "").strip().lower()
    if isinstance(payload, dict):
        modality_keys = {
            "text": ["text_extraction_confidence", "text_confidence"],
            "table": ["table_extraction_confidence", "table_confidence"],
            "image": ["image_extraction_confidence", "image_confidence", "ocr_confidence"],
        }
        for key in modality_keys.get(modality, []):
            candidates.append(payload.get(key))
        candidates.append(payload.get("extraction_confidence"))
        candidates.append(payload.get("confidence"))

    if isinstance(metadata, dict):
        meta_keys = [
            f"{modality}_extraction_confidence",
            f"{modality}_confidence",
            "default_extraction_confidence",
            "extraction_confidence",
        ]
        for key in meta_keys:
            candidates.append(metadata.get(key))

    for value in candidates:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        return max(0.0, min(1.0, numeric))

    return max(0.0, min(1.0, float(default)))


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

    image_pattern = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)")
    for line in lines:
        image_matches = image_pattern.findall(line)
        if image_matches:
            for alt_text, image_path in image_matches:
                payload = {"image_path": image_path}
                if alt_text.strip():
                    payload["alt_text"] = alt_text.strip()
                image_text = _image_to_ocr_stub(payload)
                extraction_confidence = 0.25 if image_text.startswith("[OCR_STUB]") else 0.65
                units.append(
                    ExtractedUnit(
                        text=image_text,
                        modality="image",
                        extraction_confidence=extraction_confidence,
                        metadata={
                            "image_index": image_index,
                            "image_path": image_path,
                            "alt_text": alt_text.strip(),
                        },
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
            table_text = _markdown_table_to_text(table_lines)
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
        table_text = _markdown_table_to_text(table_lines)
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


def _markdown_table_to_text(lines: Sequence[str]) -> str:
    parsed = _parse_markdown_table(lines)
    if parsed is None:
        return "\n".join(lines).strip()
    return _table_to_text(parsed)


def _parse_markdown_table(lines: Sequence[str]) -> Optional[Dict[str, Any]]:
    if len(lines) < 2:
        return None

    rows: List[List[str]] = []
    for line in lines:
        cells = _split_markdown_table_row(line)
        if cells:
            rows.append(cells)

    if len(rows) < 2:
        return None

    header = rows[0]
    separator = rows[1]
    if len(separator) != len(header):
        return None
    if not all(_is_markdown_separator_cell(cell) for cell in separator):
        return None

    if not header or len(set(header)) != len(header) or not all(header):
        return None

    structured_rows: List[Dict[str, str]] = []
    width = len(header)
    for values in rows[2:]:
        padded = values[:width] + ([""] * max(0, width - len(values)))
        structured_rows.append({header[idx]: padded[idx] for idx in range(width)})
    return {"rows": structured_rows}


def _split_markdown_table_row(line: str) -> List[str]:
    stripped = line.strip()
    if not stripped.startswith("|") or "|" not in stripped:
        return []
    if stripped.startswith("|"):
        stripped = stripped[1:]
    if stripped.endswith("|"):
        stripped = stripped[:-1]
    return [cell.strip() for cell in stripped.split("|")]


def _is_markdown_separator_cell(cell: str) -> bool:
    if not cell:
        return False
    reduced = cell.replace(" ", "")
    if len(reduced) < 3:
        return False
    if set(reduced) <= {"-"}:
        return True
    if reduced.startswith(":") and reduced.endswith(":"):
        return set(reduced[1:-1]) <= {"-"}
    if reduced.startswith(":"):
        return set(reduced[1:]) <= {"-"}
    if reduced.endswith(":"):
        return set(reduced[:-1]) <= {"-"}
    return False


def _infer_repository_modality(extension: str) -> str:
    if extension in _REPO_CODE_EXTENSIONS:
        return "code"
    if extension in {".md", ".markdown", ".mdx", ".rst"}:
        return "markdown_mixed"
    if extension in {".csv", ".tsv"}:
        return "table"
    return "text"


_REPO_CODE_EXTENSIONS = {".py", ".js", ".ts", ".jsx", ".tsx", ".java", ".go", ".rs"}


def _build_repository_manifest_documents(file_records: Sequence[Dict[str, Any]]) -> List[IngestDocument]:
    if not file_records:
        return []

    extension_counts: Dict[str, int] = {}
    directory_counts: Dict[str, int] = {}
    file_lines: List[str] = []

    for record in sorted(file_records, key=lambda item: str(item.get("rel_path", ""))):
        rel_path = str(record.get("rel_path", "")).strip()
        if not rel_path:
            continue
        extension = str(record.get("extension", "")).lower()
        content = str(record.get("content", ""))
        modality = str(record.get("modality", "text")).strip() or "text"

        extension_counts[extension] = extension_counts.get(extension, 0) + 1
        directory = Path(rel_path).parent.as_posix()
        directory_counts[directory] = directory_counts.get(directory, 0) + 1

        symbols = _extract_repository_symbols(rel_path=rel_path, extension=extension, content=content)
        symbol_text = ", ".join(symbols[:8]) if symbols else "none"
        file_lines.append(
            f"- file={rel_path} ext={extension or 'none'} modality={modality} symbols={symbol_text}"
        )

    if not file_lines:
        return []

    top_extensions = sorted(extension_counts.items(), key=lambda item: (-item[1], item[0]))
    top_directories = sorted(directory_counts.items(), key=lambda item: (-item[1], item[0]))
    extension_summary = ", ".join(f"{ext or '[no_ext]'}:{count}" for ext, count in top_extensions[:8])
    directory_summary = ", ".join(f"{directory}:{count}" for directory, count in top_directories[:10])

    overview_lines = [
        "Repository manifest overview",
        f"total_files={len(file_lines)}",
        f"extension_distribution={extension_summary}",
        f"directory_distribution={directory_summary}",
        "Use this index as a fallback for codewiki-style repository understanding.",
    ]
    overview = "\n".join(overview_lines)

    catalog_header = [
        "Repository file inventory",
        "Each line captures file path, modality, and detected symbols/headings.",
    ]
    catalog = "\n".join(catalog_header + file_lines[:300])

    return [
        IngestDocument(
            source_id="repo:__manifest__/overview",
            source_type=SourceType.REPOSITORY,
            content=overview,
            modality="text",
            metadata={
                "manifest_type": "repo_overview",
                "extraction_confidence": 0.9,
            },
        ),
        IngestDocument(
            source_id="repo:__manifest__/files",
            source_type=SourceType.REPOSITORY,
            content=catalog,
            modality="text",
            metadata={
                "manifest_type": "repo_file_inventory",
                "extraction_confidence": 0.88,
            },
        ),
    ]


def _extract_repository_symbols(rel_path: str, extension: str, content: str) -> List[str]:
    if extension in _REPO_CODE_EXTENSIONS:
        language = detect_language(rel_path, content)
        code_units = extract_code_units(content, language)
        symbols = [
            f"{unit.unit_type}:{unit.name}"
            for unit in code_units
            if unit.unit_type in {"class", "function", "method", "constant"}
        ]
        return _dedupe(symbols)[:20]

    if extension in {".md", ".markdown", ".mdx", ".rst"}:
        headings = re.findall(r"^\s*#{1,6}\s+(.+)$", content, re.MULTILINE)
        normalized = [f"heading:{heading.strip()}" for heading in headings if heading.strip()]
        return _dedupe(normalized)[:20]

    return []


def _parse_delimited_table(content: str, extension: str) -> Dict[str, Any]:
    delimiter = "," if extension == ".csv" else "\t"
    rows: List[List[str]] = []
    for row in csv.reader(content.splitlines(), delimiter=delimiter):
        cleaned = [cell.strip() for cell in row]
        if not any(cleaned):
            continue
        rows.append(cleaned)

    if not rows:
        return {"rows": []}

    header = rows[0]
    body = rows[1:]
    if body and all(header) and len(set(header)) == len(header):
        structured_rows: List[Dict[str, str]] = []
        width = len(header)
        for values in body:
            padded = values[:width] + ([""] * max(0, width - len(values)))
            structured_rows.append({header[idx]: padded[idx] for idx in range(width)})
        return {"rows": structured_rows}

    return {"rows": rows}


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


def _source_reliability(chunk: Chunk) -> float:
    """Reliability signal for ingestion provenance and extraction path quality."""
    reliability = (_source_authority(chunk) * 0.60) + (_extraction_quality(chunk) * 0.40)

    if chunk.metadata.get("origin_path") or chunk.metadata.get("path"):
        reliability += 0.04
    if chunk.metadata.get("unit_kind") in {"text", "table", "image"}:
        reliability += 0.03
    if chunk.metadata.get("chunk_type") == "full_unit":
        reliability += 0.02

    if chunk.modality == "image_ocr_stub":
        reliability -= 0.22

    route_quality = _ingestion_route_quality(chunk)
    reliability = (reliability * 0.84) + (route_quality * 0.16)

    return max(0.0, min(1.0, reliability))


def _ingestion_route_quality(chunk: Chunk) -> float:
    route = str(chunk.metadata.get("ingestion_route", "")).strip().lower()
    route_scores = {
        "pipeline_verified": 0.98,
        "repository_scan": 0.94,
        "artifact_bundle": 0.90,
        "normalized_record": 0.84,
        "explicit_record": 0.82,
        "pipeline_ocr_stub": 0.40,
        "ocr_stub": 0.40,
    }
    if route:
        return max(0.0, min(1.0, route_scores.get(route, 0.76)))
    if chunk.modality == "image_ocr_stub":
        return 0.35
    return 0.88


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

    # 5. Source reliability/provenance quality
    reliability = _source_reliability(chunk)

    # 6. Chunk completeness
    completeness = _chunk_completeness(chunk)

    # 7. Rank-based decay (lower ranks get slight penalty)
    rank_decay = 1.0 - (rank * 0.08)

    # 8. Score-based confidence from retrieval
    base_score = score_breakdown.get("lexical", 0.0) + score_breakdown.get("semantic", 0.0)
    score_confidence = min(1.0, base_score * 2)

    # Weighted combination
    confidence = (
        coverage * 0.20 +
        extraction * 0.14 +
        position * 0.08 +
        authority * 0.11 +
        reliability * 0.18 +
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


def _cross_source_consensus(ranked: Sequence[RankedChunk]) -> float:
    if not ranked:
        return 0.0

    term_support: Dict[str, set] = {}
    for item in ranked:
        for term in set(item.matched_terms):
            term_support.setdefault(term, set()).add(item.chunk.source_id)

    if not term_support:
        return 0.0

    corroborated = sum(1 for sources in term_support.values() if len(sources) >= 2)
    return max(0.0, min(1.0, corroborated / len(term_support)))


def _source_concentration(ranked: Sequence[RankedChunk]) -> float:
    if not ranked:
        return 0.0

    by_source: Dict[str, int] = {}
    for item in ranked:
        by_source[item.chunk.source_id] = by_source.get(item.chunk.source_id, 0) + 1

    max_share = max(by_source.values()) / len(ranked)
    return max(0.0, min(1.0, max_share))


def _cross_source_conflict(ranked: Sequence[RankedChunk]) -> float:
    if len(ranked) < 2:
        return 0.0

    comparable_pairs = 0
    conflicting_pairs = 0
    token_cache: Dict[str, set] = {}

    for idx, left in enumerate(ranked):
        left_key = left.chunk.chunk_id
        left_tokens = token_cache.setdefault(left_key, set(_tokenize(left.chunk.text)))
        for right in ranked[idx + 1:]:
            if left.chunk.source_id == right.chunk.source_id:
                continue
            right_key = right.chunk.chunk_id
            right_tokens = token_cache.setdefault(right_key, set(_tokenize(right.chunk.text)))
            shared_context = (
                left_tokens.intersection(right_tokens)
                .difference(_CONFLICT_TERMS)
                .difference(_CONFLICT_CONTEXT_STOPWORDS)
            )
            if not shared_context:
                continue

            comparable_pairs += 1
            has_conflict = False
            for positive, negative in _CONFLICT_POLARITY_PAIRS:
                if (positive in left_tokens and negative in right_tokens) or (
                    negative in left_tokens and positive in right_tokens
                ):
                    has_conflict = True
                    break
            if has_conflict:
                conflicting_pairs += 1

    if comparable_pairs == 0:
        return 0.0
    return max(0.0, min(1.0, conflicting_pairs / comparable_pairs))


def _calibrate_retrieval_confidence(
    ranked: Sequence[RankedChunk],
    plan: QueryPlan,
    aggregate_confidence: float,
    missing_sources: Sequence[SourceType],
    missing_modalities: Sequence[str],
    unavailable_sources: Sequence[SourceType],
    unavailable_modalities: Sequence[str],
) -> Tuple[float, Dict[str, float]]:
    preferred_sources = len(plan.preferred_source_types)
    preferred_modalities = len(plan.preferred_modalities)
    source_coverage = 1.0 - (len(missing_sources) / preferred_sources) if preferred_sources else 1.0
    modality_coverage = 1.0 - (len(missing_modalities) / preferred_modalities) if preferred_modalities else 1.0

    if ranked:
        ocr_stub_count = sum(1 for item in ranked if item.chunk.modality == "image_ocr_stub")
        low_conf_count = sum(1 for item in ranked if item.confidence < 0.35)
        ocr_stub_ratio = ocr_stub_count / len(ranked)
        low_signal_ratio = low_conf_count / len(ranked)
        cross_source_consensus = _cross_source_consensus(ranked)
        cross_source_conflict = _cross_source_conflict(ranked)
        source_concentration = _source_concentration(ranked)
        extraction_reliability = sum(_extraction_quality(item.chunk) for item in ranked) / len(ranked)
        source_reliability = sum(_source_reliability(item.chunk) for item in ranked) / len(ranked)
        ingestion_route_quality = sum(_ingestion_route_quality(item.chunk) for item in ranked) / len(ranked)
    else:
        ocr_stub_ratio = 0.0
        low_signal_ratio = 1.0
        cross_source_consensus = 0.0
        cross_source_conflict = 0.0
        source_concentration = 0.0
        extraction_reliability = 0.0
        source_reliability = 0.0
        ingestion_route_quality = 0.0

    unavailable_pressure = 0.0
    if preferred_sources:
        unavailable_pressure += (len(unavailable_sources) / preferred_sources) * 0.6
    if preferred_modalities:
        unavailable_pressure += (len(unavailable_modalities) / preferred_modalities) * 0.4
    unavailable_pressure = max(0.0, min(1.0, unavailable_pressure))

    coverage_multiplier = (0.56 + (0.24 * source_coverage) + (0.20 * modality_coverage))
    consensus_multiplier = 0.96 + (0.04 * cross_source_consensus)
    extraction_multiplier = 0.92 + (0.08 * extraction_reliability)
    reliability_multiplier = 0.90 + (0.10 * source_reliability)
    route_multiplier = 0.92 + (0.08 * ingestion_route_quality)
    quality_penalty = (0.18 * ocr_stub_ratio) + (0.12 * low_signal_ratio) + (0.16 * cross_source_conflict)
    availability_penalty = 0.08 * unavailable_pressure
    concentration_penalty = 0.04 * max(0.0, source_concentration - 0.75)

    calibrated = (
        aggregate_confidence
        * coverage_multiplier
        * consensus_multiplier
        * extraction_multiplier
        * reliability_multiplier
        * route_multiplier
    )
    calibrated = calibrated * max(0.55, 1.0 - quality_penalty - availability_penalty - concentration_penalty)
    calibrated = max(0.0, min(1.0, calibrated))

    factors = {
        "source_coverage": round(max(0.0, min(1.0, source_coverage)), 4),
        "modality_coverage": round(max(0.0, min(1.0, modality_coverage)), 4),
        "ocr_stub_ratio": round(max(0.0, min(1.0, ocr_stub_ratio)), 4),
        "low_signal_ratio": round(max(0.0, min(1.0, low_signal_ratio)), 4),
        "unavailable_pressure": round(unavailable_pressure, 4),
        "cross_source_consensus": round(max(0.0, min(1.0, cross_source_consensus)), 4),
        "cross_source_conflict": round(max(0.0, min(1.0, cross_source_conflict)), 4),
        "source_concentration": round(max(0.0, min(1.0, source_concentration)), 4),
        "extraction_reliability": round(max(0.0, min(1.0, extraction_reliability)), 4),
        "source_reliability": round(max(0.0, min(1.0, source_reliability)), 4),
        "ingestion_route_quality": round(max(0.0, min(1.0, ingestion_route_quality)), 4),
    }
    return round(calibrated, 4), factors


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
                "reliability_weighted": 0.0,
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
        entry["reliability_weighted"] += _source_reliability(item.chunk) * weight
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
                reliability_score=round(entry["reliability_weighted"] / weight_sum, 4),
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


def _should_apply_recovery(
    aggregate_confidence: float,
    missing_sources: Sequence[SourceType],
    missing_modalities: Sequence[str],
) -> bool:
    if aggregate_confidence < 0.55:
        return True
    if missing_modalities:
        return True
    if len(missing_sources) >= 2:
        return True
    return False


def _build_recovery_queries(
    query_text: str,
    plan: QueryPlan,
    missing_sources: Sequence[SourceType],
    missing_modalities: Sequence[str],
    max_queries: int = 3,
) -> List[str]:
    queries: List[str] = []

    if "table" in missing_modalities:
        queries.append(f"{query_text} table matrix rows columns evidence")
    if "image" in missing_modalities or "image_ocr_stub" in missing_modalities:
        queries.append(f"{query_text} image screenshot diagram ocr evidence")

    source_hint = {
        SourceType.REQUIREMENTS: "requirements acceptance criteria expected behavior",
        SourceType.SYSTEM_ANALYSIS: "system analysis architecture diagnostics failure modes",
        SourceType.CODE_SNIPPET: "code snippet function logic edge case branch handling",
        SourceType.REPOSITORY: "repository module implementation traceback call path",
        SourceType.KNOWLEDGE: "background knowledge prior incidents mitigations",
    }
    for source_type in missing_sources[:2]:
        hint = source_hint.get(source_type)
        if hint:
            queries.append(f"{query_text} {hint}")

    if not queries:
        intent_terms = " ".join(plan.intent_labels) if plan.intent_labels else "test analysis"
        queries.append(f"{query_text} {intent_terms} evidence")

    deduped: List[str] = []
    seen = set()
    for query in queries:
        if query in seen:
            continue
        seen.add(query)
        deduped.append(query)
        if len(deduped) >= max(1, max_queries):
            break
    return deduped


def _build_focus_queries(query_text: str) -> List[Tuple[str, str]]:
    base = query_text.strip() or "test analysis"
    focus_hints = [
        ("failure_clustering", "cluster recurring flaky failure patterns and shared signatures"),
        ("root_cause", "root cause traceback hypothesis and failure origin"),
        ("test_gap", "missing test coverage negative and edge case scenarios"),
        ("risk_prioritization", "release risk severity prioritization p0 blocking issues"),
        ("actionable_plan", "actionable mitigation plan next steps and owner assignments"),
    ]
    return [(focus, f"{base} {hint}".strip()) for focus, hint in focus_hints]


def _recommend_ingestion_actions(
    plan: QueryPlan,
    corpus_profile: CorpusCoverageProfile,
    missing_sources: Sequence[SourceType],
    missing_modalities: Sequence[str],
    unavailable_sources: Sequence[SourceType],
    unavailable_modalities: Sequence[str],
    max_actions: int = 5,
) -> List[str]:
    actions: List[str] = []

    if "table" in unavailable_modalities:
        actions.append(
            "Add tabular artifacts (CSV/TSV/markdown tables) from incident reports or requirements to improve risk matrix evidence."
        )
    if "image" in unavailable_modalities or "image_ocr_stub" in unavailable_modalities:
        actions.append(
            "Connect OCR extraction for screenshots/diagrams and ingest image metadata (path, alt text, OCR confidence)."
        )

    source_prompts = {
        SourceType.REPOSITORY: "Ingest targeted repository modules covering failing call paths and traceback locations.",
        SourceType.SYSTEM_ANALYSIS: "Ingest system analysis docs (architecture, telemetry notes, incident diagnostics) for root-cause context.",
        SourceType.CODE_SNIPPET: "Add focused code snippets around failing functions to improve precise root-cause hypotheses.",
        SourceType.REQUIREMENTS: "Ingest requirement and acceptance-criteria docs for stronger test-gap and priority mapping.",
        SourceType.KNOWLEDGE: "Ingest background knowledge (runbooks, prior incidents) for mitigation pattern retrieval.",
    }
    for source in unavailable_sources[:3]:
        suggestion = source_prompts.get(source)
        if suggestion:
            actions.append(suggestion)

    low_quality_sources = sorted(
        (
            source_type
            for source_type, avg in corpus_profile.source_type_avg_extraction.items()
            if avg < 0.5 and source_type in plan.preferred_source_types
        ),
        key=lambda stype: corpus_profile.source_type_avg_extraction.get(stype, 1.0),
    )
    if low_quality_sources:
        source_names = ", ".join(source.value for source in low_quality_sources[:2])
        actions.append(
            f"Improve extraction quality for {source_names} (higher OCR/table parser confidence and better metadata provenance)."
        )

    if missing_modalities and not unavailable_modalities:
        actions.append(
            "Tune retrieval weights/query expansion for under-retrieved modalities already present in the corpus."
        )
    if missing_sources and not unavailable_sources:
        actions.append(
            "Tune retrieval diversification to surface existing but under-selected source types for this intent."
        )

    deduped: List[str] = []
    seen = set()
    for action in actions:
        key = action.strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(action)
        if len(deduped) >= max(1, max_actions):
            break
    return deduped


def _fuse_ranked_chunks(primary: Sequence[RankedChunk], supplemental: Sequence[RankedChunk]) -> List[RankedChunk]:
    merged: Dict[str, RankedChunk] = {}
    recovery_hits: Dict[str, int] = {}

    for item in primary:
        merged[item.chunk.chunk_id] = RankedChunk(
            chunk=item.chunk,
            score=item.score,
            confidence=item.confidence,
            matched_terms=list(item.matched_terms),
            score_breakdown=dict(item.score_breakdown),
        )
        recovery_hits[item.chunk.chunk_id] = 0

    for item in supplemental:
        key = item.chunk.chunk_id
        recovery_hits[key] = recovery_hits.get(key, 0) + 1
        if key not in merged:
            merged[key] = RankedChunk(
                chunk=item.chunk,
                score=item.score,
                confidence=item.confidence,
                matched_terms=list(item.matched_terms),
                score_breakdown=dict(item.score_breakdown),
            )
            continue
        existing = merged[key]
        existing.score = max(existing.score, item.score)
        existing.confidence = max(existing.confidence, item.confidence)
        existing_terms = set(existing.matched_terms)
        existing_terms.update(item.matched_terms)
        existing.matched_terms = sorted(existing_terms)
        for name, value in item.score_breakdown.items():
            existing.score_breakdown[name] = max(existing.score_breakdown.get(name, 0.0), value)

    fused: List[RankedChunk] = []
    for key, item in merged.items():
        hits = recovery_hits.get(key, 0)
        if hits > 0:
            item.score += min(0.06, 0.02 * hits)
            item.confidence = round(max(0.0, min(1.0, item.confidence + min(0.08, 0.02 * hits))), 4)
            item.score_breakdown["recovery_hits"] = float(hits)
        fused.append(item)

    fused.sort(
        key=lambda candidate: (
            -candidate.score,
            -len(candidate.matched_terms),
            candidate.chunk.source_id,
            candidate.chunk.chunk_id,
        )
    )
    return fused


def _apply_corroboration_bonus(candidates: Sequence[RankedChunk], max_bonus: float = 0.10) -> None:
    if not candidates:
        return

    term_sources: Dict[str, set] = {}
    term_source_types: Dict[str, set] = {}
    term_modalities: Dict[str, set] = {}

    for item in candidates:
        for term in set(item.matched_terms):
            term_sources.setdefault(term, set()).add(item.chunk.source_id)
            term_source_types.setdefault(term, set()).add(item.chunk.source_type)
            term_modalities.setdefault(term, set()).add(item.chunk.modality)

    for item in candidates:
        if not item.matched_terms:
            item.score_breakdown["corroboration"] = 0.0
            continue

        term_scores: List[float] = []
        for term in set(item.matched_terms):
            support_sources = max(0, len(term_sources.get(term, set())) - 1)
            support_types = max(0, len(term_source_types.get(term, set())) - 1)
            support_modalities = max(0, len(term_modalities.get(term, set())) - 1)
            corroboration = min(
                1.0,
                (support_sources * 0.22) + (support_types * 0.40) + (support_modalities * 0.20),
            )
            term_scores.append(corroboration)

        aggregate = sum(term_scores) / max(len(term_scores), 1)
        breadth = min(1.0, len(set(item.matched_terms)) / 3.0)
        aggregate *= breadth
        bonus = min(max_bonus, aggregate * max_bonus)
        item.score += bonus
        item.score_breakdown["corroboration"] = round(aggregate, 4)


def _select_coverage_aware_top(
    candidates: Sequence[RankedChunk],
    plan: QueryPlan,
    top_k: int,
    diversify: bool,
) -> List[RankedChunk]:
    if top_k <= 0 or not candidates:
        return []
    if len(candidates) <= top_k:
        return list(candidates)

    selected: List[RankedChunk] = []
    remaining = list(candidates)
    needed_modalities = set(mod for mod in plan.preferred_modalities if mod != "text")
    available_source_types = {candidate.chunk.source_type for candidate in candidates}
    preferred_available = [stype for stype in plan.preferred_source_types if stype in available_source_types]
    source_target_budget = min(len(preferred_available), max(1, min(top_k, 3)))
    needed_sources = set(preferred_available[:source_target_budget])
    source_counts: Dict[str, int] = {}
    max_per_source = max(1, top_k - 1)

    while remaining and len(selected) < top_k:
        best_idx = 0
        best_value = float("-inf")
        selected_source_ids = {item.chunk.source_id for item in selected}
        for idx, candidate in enumerate(remaining):
            value = candidate.score
            if candidate.chunk.modality in needed_modalities:
                value += 0.24
            if candidate.chunk.source_type in needed_sources:
                value += 0.10
            if diversify and candidate.chunk.source_id not in selected_source_ids:
                value += 0.22
            if selected:
                similarity = max(_chunk_similarity(candidate.chunk, existing.chunk) for existing in selected)
                value -= (0.12 * similarity)
            if diversify:
                source_id = candidate.chunk.source_id
                if source_counts.get(source_id, 0) >= max_per_source:
                    alternatives = any(item.chunk.source_id != source_id for item in remaining)
                    if alternatives:
                        value -= 0.40
            if value > best_value:
                best_value = value
                best_idx = idx
        picked = remaining.pop(best_idx)
        selected.append(picked)
        source_counts[picked.chunk.source_id] = source_counts.get(picked.chunk.source_id, 0) + 1
        needed_modalities.discard(picked.chunk.modality)
        needed_sources.discard(picked.chunk.source_type)

    if diversify:
        selected.sort(
            key=lambda candidate: (
                -candidate.score,
                -len(candidate.matched_terms),
                candidate.chunk.source_id,
                candidate.chunk.chunk_id,
            )
        )
    return selected[:top_k]


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

    def prepare_corpus(self, texts: List[str]) -> List[List[float]]:
        """Fit any corpus-specific state and return corpus embeddings."""
        return self.encode(texts)

    def encode_query(self, text: str) -> List[float]:
        """Encode query text in the same vector space as corpus embeddings."""
        vectors = self.encode([text])
        return vectors[0] if vectors else []


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

        # Preserve backwards compatibility for corpus encoding while avoiding
        # accidental vocabulary resets for single query encodes.
        if len(texts) > 1 or not self._vocabulary:
            return self.prepare_corpus(texts)

        # Encode each text
        vectors: List[List[float]] = []
        for text in texts:
            vectors.append(self._text_to_vector(text))

        return vectors

    def prepare_corpus(self, texts: List[str]) -> List[List[float]]:
        """Fit vocabulary/IDF on corpus texts and return corpus vectors."""
        if not texts:
            return []
        self._build_vocabulary(texts)
        return [self._text_to_vector(text) for text in texts]

    def encode_query(self, text: str) -> List[float]:
        """Encode a query using the fitted vocabulary if available."""
        if not self._vocabulary:
            vectors = self.prepare_corpus([text])
            return vectors[0] if vectors else []
        return self._text_to_vector(text)

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

    def ingest_documents(
        self,
        docs: Sequence[IngestDocument],
        generate_source_summaries: bool = False,
    ) -> List[Chunk]:
        """Ingest documents and compute embeddings."""
        chunks = super().ingest_documents(
            docs,
            generate_source_summaries=generate_source_summaries,
        )
        self._compute_embeddings()
        return chunks

    def _compute_embeddings(self) -> None:
        """Compute embeddings for all chunks."""
        if not self._chunks:
            return
        texts = [_embedding_projection_text(chunk) for chunk in self._chunks]
        try:
            embeddings = self._embedding_provider.prepare_corpus(texts)
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
                query_embedding = self._embedding_provider.encode_query(query_text)
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
                    + extraction_quality * 0.06
                    + _source_reliability(chunk) * 0.04
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
                    + extraction_quality * 0.06
                    + _source_reliability(chunk) * 0.04
                )

            breakdown = {
                "lexical": round(lex_score, 4),
                "semantic": round(sem_score, 4) if use_hybrid else 0.0,
                "source": round(source_boost, 4),
                "intent": round(intent_boost, 4),
                "modality": round(modality_boost, 4),
                "extraction": round(extraction_quality, 4),
                "reliability": round(_source_reliability(chunk), 4),
                "position": round(_position_score(chunk), 4),
                "authority": round(_source_authority(chunk), 4),
                "completeness": round(_chunk_completeness(chunk, avg_chunk_size=self._chunk_size), 4),
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

        _apply_corroboration_bonus(scored)
        _apply_corroboration_bonus(fallback)

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
            enhanced_confidence = compute_enhanced_confidence(
                chunk=item.chunk,
                query_tokens=plan.tokens,
                matched_terms=item.matched_terms,
                score_breakdown=item.score_breakdown,
                rank=rank,
                total_results=len(top),
            )
            normalized = item.score / max(max_score, 1e-9)
            coverage = len(item.matched_terms) / max(len(plan.tokens), 1)
            extraction = item.score_breakdown.get("extraction", 1.0)
            decay = 1.0 - (rank * 0.12)
            legacy_confidence = (
                (normalized * 0.50)
                + (coverage * 0.22)
                + (extraction * 0.18)
                + (diversity_bonus * 0.10)
            )
            blended = (enhanced_confidence * 0.72) + (legacy_confidence * decay * 0.28)
            item.confidence = round(max(0.0, min(1.0, blended)), 4)

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


def _embedding_projection_text(chunk: Chunk) -> str:
    """Create a semantically useful text projection for embedding models."""
    text = chunk.text
    if chunk.modality == "image_ocr_stub":
        text = text.replace("[OCR_STUB]", " ")
        text = text.replace("no OCR pipeline connected for", " ")
        text = text.replace("alt_text:", " ")

    descriptors = [
        f"modality {chunk.modality}",
        f"source_type {chunk.source_type.value}",
    ]
    unit_kind = chunk.metadata.get("unit_kind")
    if unit_kind:
        descriptors.append(f"unit_kind {unit_kind}")
    path = chunk.metadata.get("path") or chunk.metadata.get("origin_path")
    if path:
        descriptors.append(f"path {path}")

    return " ".join(descriptors + [text]).strip()


def _build_source_summary_chunks(chunks: Sequence[Chunk]) -> List[Chunk]:
    grouped: Dict[Tuple[str, SourceType], List[Chunk]] = {}
    for chunk in chunks:
        if chunk.metadata.get("synthetic_unit") == "source_summary":
            continue
        if chunk.source_id.endswith("::__summary__"):
            continue
        grouped.setdefault((chunk.source_id, chunk.source_type), []).append(chunk)

    summary_chunks: List[Chunk] = []
    for (source_id, source_type), source_chunks in grouped.items():
        modalities = sorted({chunk.modality for chunk in source_chunks})
        if len(source_chunks) < 2 and len(modalities) < 2:
            continue

        token_counts: Dict[str, int] = {}
        for chunk in source_chunks:
            for token in _tokenize(chunk.text):
                if len(token) < 3:
                    continue
                token_counts[token] = token_counts.get(token, 0) + 1
        top_terms = [term for term, _ in sorted(token_counts.items(), key=lambda item: (-item[1], item[0]))[:10]]

        sample_snippets: List[str] = []
        for chunk in source_chunks:
            snippet = " ".join(chunk.text.strip().split())
            if not snippet:
                continue
            if len(snippet) > 120:
                snippet = snippet[:117] + "..."
            sample_snippets.append(f"{chunk.modality}:{snippet}")
            if len(sample_snippets) >= 3:
                break

        avg_extraction = sum(_extraction_quality(chunk) for chunk in source_chunks) / max(len(source_chunks), 1)
        summary_source_id = f"{source_id}::__summary__"
        summary_lines = [
            f"source_summary source_id={source_id}",
            f"source_type={source_type.value} modalities={','.join(modalities)} chunks={len(source_chunks)}",
            f"avg_extraction_quality={avg_extraction:.2f}",
        ]
        if top_terms:
            summary_lines.append(f"top_terms={','.join(top_terms)}")
        if sample_snippets:
            summary_lines.append("highlights=" + " || ".join(sample_snippets))
        summary_text = "\n".join(summary_lines)

        summary_chunks.append(
            Chunk(
                chunk_id=_hash_id(summary_source_id, 0, 0, summary_text),
                source_id=summary_source_id,
                source_type=source_type,
                modality="text",
                text=summary_text,
                token_count=len(_tokenize(summary_text)),
                metadata={
                    "synthetic_unit": "source_summary",
                    "linked_source_id": source_id,
                    "modalities": modalities,
                    "source_chunk_count": len(source_chunks),
                    "extraction_confidence": round(max(0.55, min(0.92, avg_extraction * 0.9)), 4),
                },
            )
        )

    return summary_chunks
