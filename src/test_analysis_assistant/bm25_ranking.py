"""BM25 ranking algorithm for improved lexical search.

BM25 (Best Matching 25) is a probabilistic ranking function used by search engines.
It improves over basic term overlap by:
- Term frequency saturation (diminishing returns for repeated terms)
- Document length normalization
- Better handling of rare vs common terms

This module provides a standalone BM25 implementation that can be used
as an alternative or complement to the existing hybrid retrieval.
"""

import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple


@dataclass
class BM25Stats:
    """Statistics for BM25 scoring."""

    avg_doc_length: float = 0.0
    total_docs: int = 0
    vocab_size: int = 0
    doc_count_with_term: Dict[str, int] = field(default_factory=dict)


class BM25:
    """BM25 ranking algorithm implementation.

    BM25 formula: score(Q, D) = sum over qi in Q:
        IDF(qi) * (f(qi, D) * (k1 + 1)) /
        (f(qi, D) + k1 * (1 - b + b * |D| / avgdl))

    Where:
        - f(qi, D) = term frequency of qi in document D
        - |D| = document length
        - avgdl =        - k1 average document length
 = term frequency saturation parameter (typically 1.2-2.0)
        - b = document length normalization parameter (typically 0.75)
        - IDF(qi) = inverse document frequency
    """

    # Default BM25 parameters (standard values from the literature)
    DEFAULT_K1: float = 1.5
    DEFAULT_B: float = 0.75

    def __init__(
        self,
        k1: float = DEFAULT_K1,
        b: float = DEFAULT_B,
        min_term_length: int = 1,
    ) -> None:
        """Initialize BM25 ranker.

        Args:
            k1: Term frequency saturation parameter. Higher values give
                more weight to term frequency. Typical range: 1.2-2.0
            b: Document length normalization parameter. 0.0 = no normalization,
                1.0 = full normalization. Typical value: 0.75
            min_term_length: Minimum length of terms to index
        """
        self._k1 = k1
        self._b = b
        self._min_term_length = min_term_length
        self._stats: Optional[BM25Stats] = None
        self._corpus: List[str] = []
        self._tokenized_corpus: List[List[str]] = []

    def fit(self, documents: Sequence[str]) -> "BM25":
        """Build BM25 index from documents.

        Args:
            documents: List of document texts to index

        Returns:
            Self for method chaining
        """
        self._corpus = list(documents)
        self._tokenized_corpus = [self._tokenize(doc) for doc in documents]

        # Calculate statistics
        self._stats = self._compute_stats()

        return self

    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization - split on whitespace and lowercase."""
        if not text:
            return []
        # Split on whitespace, lowercase, keep only alphanumeric
        tokens = []
        for word in text.lower().split():
            cleaned = "".join(c for c in word if c.isalnum())
            if len(cleaned) >= self._min_term_length:
                tokens.append(cleaned)
        return tokens

    def _compute_stats(self) -> BM25Stats:
        """Compute corpus statistics for BM25 scoring."""
        stats = BM25Stats()
        stats.total_docs = len(self._tokenized_corpus)

        if stats.total_docs == 0:
            return stats

        # Calculate average document length
        doc_lengths = [len(doc) for doc in self._tokenized_corpus]
        stats.avg_doc_length = sum(doc_lengths) / stats.total_docs

        # Count documents containing each term
        doc_count: Dict[str, int] = Counter()
        for doc_tokens in self._tokenized_corpus:
            unique_tokens = set(doc_tokens)
            for token in unique_tokens:
                doc_count[token] += 1

        stats.doc_count_with_term = dict(doc_count)
        stats.vocab_size = len(doc_count)

        return stats

    def _idf(self, term: str) -> float:
        """Calculate IDF for a term.

        IDF = log((N - n(q) + 0.5) / (n(q) + 0.5))

        Where:
            - N = total number of documents
            - n(q) = number of documents containing term q
        """
        if self._stats is None or self._stats.total_docs == 0:
            return 0.0

        n = self._stats.doc_count_with_term.get(term, 0)
        N = self._stats.total_docs

        # Smoothed IDF to handle unseen terms
        # Using the standard BM25 IDF formula with smoothing
        if n == 0:
            # Term not in corpus - use smoothed IDF
            # This is a common approach: IDF = log((N + 1) / (0 + 1)) = log(N + 1)
            return math.log((N + 1) / 1)

        # Standard BM25 IDF with smoothing
        return math.log((N - n + 0.5) / (n + 0.5) + 1)

    def score(self, query: str, doc_index: int) -> float:
        """Calculate BM25 score for a query against a specific document.

        Args:
            query: Query string
            doc_index: Index of the document in the corpus

        Returns:
            BM25 score (higher = better match)
        """
        if self._stats is None or doc_index >= len(self._tokenized_corpus):
            return 0.0

        query_tokens = self._tokenize(query)
        if not query_tokens:
            return 0.0

        doc_tokens = self._tokenized_corpus[doc_index]
        doc_length = len(doc_tokens)

        if doc_length == 0:
            return 0.0

        # Count term frequencies in document
        doc_tf = Counter(doc_tokens)

        score = 0.0
        for term in query_tokens:
            if term not in doc_tf:
                continue

            tf = doc_tf[term]
            idf = self._idf(term)

            # BM25 scoring formula
            numerator = tf * (self._k1 + 1)
            denominator = tf + self._k1 * (
                1 - self._b + self._b * (doc_length / self._stats.avg_doc_length)
            )

            score += idf * (numerator / denominator)

        return score

    def score_batch(self, query: str) -> List[Tuple[int, float]]:
        """Calculate BM25 scores for a query against all documents.

        Args:
            query: Query string

        Returns:
            List of (document_index, score) pairs, sorted by score descending
        """
        if self._stats is None:
            return []

        scores = []
        for i in range(len(self._tokenized_corpus)):
            s = self.score(query, i)
            if s > 0:
                scores.append((i, s))

        # Sort by score descending
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores

    def get_top_k(
        self, query: str, k: int = 10, min_score: float = 0.0
    ) -> List[Tuple[int, float]]:
        """Get top-k documents for a query.

        Args:
            query: Query string
            k: Number of results to return
            min_score: Minimum score threshold

        Returns:
            List of (document_index, score) pairs
        """
        all_scores = self.score_batch(query)
        return [(idx, score) for idx, score in all_scores if score >= min_score][:k]

    def get_scores_for_corpus(self, query: str) -> Dict[int, float]:
        """Get BM25 scores for all documents as a dictionary.

        Args:
            query: Query string

        Returns:
            Dictionary mapping document index to score
        """
        if self._stats is None:
            return {}

        return {i: self.score(query, i) for i in range(len(self._tokenized_corpus))}


class BM25Provider:
    """Provider class for using BM25 in the retrieval system.

    This class wraps BM25 and provides a simplified interface
    for integration with the existing retrieval infrastructure.
    """

    def __init__(
        self,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        """Initialize BM25 provider.

        Args:
            k1: Term frequency saturation parameter
            b: Document length normalization parameter
        """
        self._bm25 = BM25(k1=k1, b=b)
        self._fitted = False

    def fit(self, texts: Sequence[str]) -> "BM25Provider":
        """Fit BM25 provider to documents (alias for prepare_corpus).

        Args:
            texts: List of document texts

        Returns:
            Self for method chaining
        """
        return self.prepare_corpus(texts)

    def prepare_corpus(self, texts: Sequence[str]) -> "BM25Provider":
        """Prepare BM25 index from corpus.

        Args:
            texts: List of document texts

        Returns:
            Self for method chaining
        """
        self._bm25.fit(texts)
        self._fitted = True
        return self

    def get_scores(self, query: str) -> Dict[int, float]:
        """Get BM25 scores for a query.

        Args:
            query: Query string

        Returns:
            Dictionary mapping document index to BM25 score
        """
        if not self._fitted:
            return {}
        return self._bm25.get_scores_for_corpus(query)

    def get_top_results(
        self, query: str, k: int = 10
    ) -> List[Tuple[int, float]]:
        """Get top-k BM25 results.

        Args:
            query: Query string
            k: Number of results

        Returns:
            List of (document_index, score) pairs
        """
        if not self._fitted:
            return []
        return self._bm25.get_top_k(query, k=k)


def create_bm25_provider(
    k1: float = 1.5,
    b: float = 0.75,
) -> BM25Provider:
    """Create a configured BM25 provider.

    Args:
        k1: Term frequency saturation parameter
        b: Document length normalization parameter

    Returns:
        Configured BM25Provider instance
    """
    return BM25Provider(k1=k1, b=b)
