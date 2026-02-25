"""Test Analysis Assistant MVP package."""

from .analyzer import analyze_report_text
from .models import AnalysisResult, FailureRecord, FixSuggestion
from .rag_analyzer import RAGAnalyzer, RAGAnalysisResult, RetrievalInsight, rag_analyze
from .retrieval import (
    DummyEmbeddingProvider,
    HybridRetrievalEngine,
    IngestDocument,
    MultiSourceIngestor,
    RetrievalEngine,
    SourceType,
    TFIDFEmbeddingProvider,
    build_analysis_prompt,
    create_hybrid_engine,
)

__all__ = [
    "analyze_report_text",
    "AnalysisResult",
    "FailureRecord",
    "FixSuggestion",
    "IngestDocument",
    "MultiSourceIngestor",
    "RAGAnalyzer",
    "RAGAnalysisResult",
    "RetrievalInsight",
    "RetrievalEngine",
    "SourceType",
    "build_analysis_prompt",
    "rag_analyze",
    "HybridRetrievalEngine",
    "DummyEmbeddingProvider",
    "TFIDFEmbeddingProvider",
    "create_hybrid_engine",
]
