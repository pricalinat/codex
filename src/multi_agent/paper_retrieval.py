import re
import json
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set


TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
ABBREVIATION_PATTERN = re.compile(r"\b[A-Z]{2,5}\b")


@dataclass
class PaperDocument:
    paper_id: str
    title: str
    abstract: str
    keywords: List[str]


@dataclass
class RankedPaper:
    paper: PaperDocument
    score: float
    signals: Dict[str, float]


@dataclass
class BenchmarkCase:
    query: str
    goal: str
    relevant_ids: List[str]
    hard_negative_ids: Optional[List[str]] = None


@dataclass
class RetrievalMetrics:
    hit_rate_at_k: float
    mrr_at_k: float
    hard_negative_rate_at_k: float = 0.0


@dataclass
class TuningReport:
    rounds_used: int
    baseline: RetrievalMetrics
    best: RetrievalMetrics
    best_weights: Dict[str, float]


class PaperRetriever:
    def __init__(self, weights: Optional[Dict[str, float]] = None) -> None:
        self.weights = weights or {
            "title_overlap": 0.9,
            "abstract_overlap": 0.7,
            "keyword_overlap": 1.0,
            "goal_overlap": 1.3,
            "phrase_overlap": 0.8,
        }

        self._synonyms = {
            "llm": ["language", "model"],
            "nlp": ["language", "text"],
            "forecasting": ["prediction", "time", "series"],
            "transformer": ["attention"],
        }

    def add_synonym(self, abbrev: str, expansions: List[str]) -> None:
        """Add a synonym mapping for an abbreviation at runtime."""
        self._synonyms[abbrev.lower()] = expansions

    def _is_abbreviation(self, token: str) -> bool:
        """Check if a token looks like an abbreviation (all caps, 2-5 chars)."""
        if len(token) < 2 or len(token) > 5:
            return False
        return token.isupper() and token.isalpha()

    def _expand_unknown_abbreviation(
        self, token: str, papers: Sequence[PaperDocument]
    ) -> List[str]:
        """Try to expand an unknown abbreviation using document context."""
        abbrev_lower = token.lower()
        if abbrev_lower in self._synonyms:
            return self._synonyms[abbrev_lower]

        # Try to find expansion in paper titles/abstracts
        # Pattern: "Full Term (ABBR)" or "Full Term(ABBR)"
        expansion_candidates: List[str] = []
        escaped_token = re.escape(token)
        abbrev_pattern = re.compile(
            # Match expansion phrases: handles capitalized, lowercase, and hyphenated words
            # Pattern: "Full Term (ABBR)" - e.g., "Long Short-Term Memory (LSTM)", "End-to-End (E2E)"
            rf"\b([A-Za-z][A-Za-z\-']*(?:\s+[A-Za-z][A-Za-z\-']*)*)\s*\(\s*{escaped_token}\s*\)",
            re.IGNORECASE,
        )
        for paper in papers:
            for text in [paper.title, paper.abstract, " ".join(paper.keywords)]:
                matches = abbrev_pattern.findall(text)
                for match in matches:
                    if match:
                        # Use retrieval-compatible tokenizer normalization.
                        normalized_tokens = self._normalize_expansion_phrase(match.lower())
                        expansion_candidates.extend(normalized_tokens)

        if expansion_candidates:
            # Make deterministic: sort unique tokens alphabetically, return first 5
            unique_tokens = sorted(set(expansion_candidates))
            return unique_tokens[:5]

        return []


    def search(
        self,
        query: str,
        papers: Sequence[PaperDocument],
        goal: str = "",
        top_k: int = 5,
    ) -> List[RankedPaper]:
        # Extract raw-case tokens from query for abbreviation detection
        raw_query_tokens = set(ABBREVIATION_PATTERN.findall(query))
        query_tokens = self._expand_tokens(self._tokenize(query), papers, raw_query_tokens)
        query_token_list = self._tokenize_list(query)
        query_bigrams = self._ngram_set(query_token_list, 2)
        goal_tokens = self._tokenize(goal)

        ranked: List[RankedPaper] = []
        for paper in papers:
            title_tokens = self._tokenize(paper.title)
            abstract_tokens = self._tokenize(paper.abstract)
            keyword_tokens = self._tokenize(" ".join(paper.keywords))
            full_tokens = title_tokens | abstract_tokens | keyword_tokens
            document_tokens = (
                self._tokenize_list(paper.title)
                + self._tokenize_list(paper.abstract)
                + self._tokenize_list(" ".join(paper.keywords))
            )
            document_bigrams = self._ngram_set(document_tokens, 2)

            signals = {
                "title_overlap": self._overlap_ratio(query_tokens, title_tokens),
                "abstract_overlap": self._overlap_ratio(query_tokens, abstract_tokens),
                "keyword_overlap": self._overlap_ratio(query_tokens, keyword_tokens),
                "goal_overlap": self._overlap_ratio(goal_tokens, full_tokens),
                "phrase_overlap": self._overlap_ratio(query_bigrams, document_bigrams),
            }

            score = 0.0
            for name, value in signals.items():
                score += self.weights.get(name, 0.0) * value

            ranked.append(RankedPaper(paper=paper, score=score, signals=signals))

        # Sort by score descending (primary), then by paper_id ascending (tiebreaker)
        ranked.sort(key=lambda item: (-item.score, item.paper.paper_id))
        return ranked[:top_k]

    def evaluate(
        self,
        benchmark: Sequence[BenchmarkCase],
        papers: Sequence[PaperDocument],
        top_k: int = 5,
    ) -> RetrievalMetrics:
        if not benchmark:
            return RetrievalMetrics(hit_rate_at_k=0.0, mrr_at_k=0.0, hard_negative_rate_at_k=0.0)

        hits = 0
        mrr_total = 0.0
        hard_negative_hits = 0
        for case in benchmark:
            ranked = self.search(case.query, papers, goal=case.goal, top_k=top_k)
            relevant: Set[str] = set(case.relevant_ids)
            hard_negative: Set[str] = set(case.hard_negative_ids or [])
            first_rank = 0

            for index, item in enumerate(ranked, start=1):
                if item.paper.paper_id in relevant:
                    first_rank = index
                    break
            if any(item.paper.paper_id in hard_negative for item in ranked):
                hard_negative_hits += 1

            if first_rank > 0:
                hits += 1
                mrr_total += 1.0 / float(first_rank)

        total = float(len(benchmark))
        return RetrievalMetrics(
            hit_rate_at_k=hits / total,
            mrr_at_k=mrr_total / total,
            hard_negative_rate_at_k=hard_negative_hits / total,
        )

    def tune(
        self,
        benchmark: Sequence[BenchmarkCase],
        papers: Sequence[PaperDocument],
        top_k: int = 5,
        max_rounds: int = 100,
        target_hit_rate: float = 0.9,
    ) -> TuningReport:
        baseline = self.evaluate(benchmark, papers, top_k=top_k)
        best = baseline
        best_weights = dict(self.weights)

        step = 0.2
        rounds = 0
        keys = ["title_overlap", "abstract_overlap", "keyword_overlap", "goal_overlap", "phrase_overlap"]
        delta_multipliers = (8.0, 4.0, 2.0, 1.0, -1.0, -2.0, -4.0, -8.0)

        while rounds < max_rounds:
            rounds += 1
            improved = False

            for key in keys:
                for multiplier in delta_multipliers:
                    delta = step * multiplier
                    candidate = dict(best_weights)
                    candidate[key] = max(0.0, candidate.get(key, 0.0) + delta)

                    self.weights = candidate
                    metric = self.evaluate(benchmark, papers, top_k=top_k)

                    if self._is_better(metric, best):
                        best = metric
                        best_weights = candidate
                        improved = True

            self.weights = dict(best_weights)

            if best.hit_rate_at_k >= target_hit_rate:
                break

            if not improved:
                step /= 2.0
                if step < 0.01:
                    break

        self.weights = dict(best_weights)
        return TuningReport(rounds_used=rounds, baseline=baseline, best=best, best_weights=best_weights)

    @staticmethod
    def _is_better(left: RetrievalMetrics, right: RetrievalMetrics) -> bool:
        if left.hit_rate_at_k != right.hit_rate_at_k:
            return left.hit_rate_at_k > right.hit_rate_at_k
        if left.hard_negative_rate_at_k != right.hard_negative_rate_at_k:
            return left.hard_negative_rate_at_k < right.hard_negative_rate_at_k
        return left.mrr_at_k > right.mrr_at_k

    def _expand_tokens(self, tokens: Set[str], papers: Sequence[PaperDocument], raw_tokens: Optional[Set[str]] = None) -> Set[str]:
        """Expand tokens with known synonyms and try to expand unknown abbreviations.
        
        Args:
            tokens: Lowercased tokens for synonym matching
            papers: Document corpus for abbreviation expansion
            raw_tokens: Optional raw-case tokens from original query for abbreviation detection
        """
        expanded = set(tokens)
        for token in list(tokens):
            # Known synonyms (use lowercased tokens)
            for synonym in self._synonyms.get(token, []):
                expanded.add(synonym)
            # Unknown abbreviation handling (use raw-case for is_abbreviation check)
            # Find the corresponding raw token for this lowercase token
            raw_token = None
            if raw_tokens is not None:
                for rt in raw_tokens:
                    if rt.lower() == token:
                        raw_token = rt
                        break
            check_token = raw_token if raw_token else token
            if self._is_abbreviation(check_token) and check_token.lower() not in self._synonyms:
                expansions = self._expand_unknown_abbreviation(check_token, papers)
                for exp in expansions:
                    expanded.add(exp)
        return expanded

    @staticmethod
    def _tokenize(text: str) -> Set[str]:
        if not text:
            return set()
        return set(PaperRetriever._tokenize_list(text))

    @staticmethod
    def _tokenize_list(text: str) -> List[str]:
        if not text:
            return []
        return [PaperRetriever._normalize_token(token) for token in TOKEN_PATTERN.findall(text.lower())]

    @staticmethod
    def _normalize_token(token: str) -> str:
        # lightweight stemming for common scientific morphology variants
        suffixes = [
            "ability",
            "ibility",
            "ation",
            "ition",
            "able",
            "ible",
            "ment",
            "ness",
            "tion",
            "sion",
            "ity",
            "ing",
            "ies",
            "ied",
            "ed",
            "es",
            "s",
        ]

        stemmed = token
        for suffix in suffixes:
            if stemmed.endswith(suffix) and len(stemmed) > len(suffix) + 2:
                if suffix == "ies":
                    stemmed = stemmed[: -len(suffix)] + "y"
                elif suffix == "ied":
                    stemmed = stemmed[: -len(suffix)] + "y"
                else:
                    stemmed = stemmed[: -len(suffix)]
                break

        return stemmed

    @staticmethod
    def _normalize_expansion_phrase(phrase: str) -> List[str]:
        """Normalize expansion phrase using the same tokenizer as documents/queries."""
        if not phrase:
            return []
        return PaperRetriever._tokenize_list(phrase)

    @staticmethod
    def _overlap_ratio(left: Set[str], right: Set[str]) -> float:
        if not left or not right:
            return 0.0
        inter = len(left & right)
        return float(inter) / float(len(left))

    @staticmethod
    def _ngram_set(tokens: List[str], n: int) -> Set[str]:
        if len(tokens) < n:
            return set()
        return {" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}


class TwoStagePaperRetriever:
    def __init__(self, base_retriever: Optional[PaperRetriever] = None, rerank_top_n: int = 10) -> None:
        self.base_retriever = base_retriever or PaperRetriever()
        self.rerank_top_n = rerank_top_n

    def search(
        self,
        query: str,
        papers: Sequence[PaperDocument],
        goal: str = "",
        top_k: int = 5,
    ) -> List[RankedPaper]:
        candidate_count = max(top_k, self.rerank_top_n)
        stage_one = self.base_retriever.search(query=query, papers=papers, goal=goal, top_k=candidate_count)
        if not goal:
            return stage_one[:top_k]

        goal_tokens = self.base_retriever._tokenize(goal)
        rescored: List[RankedPaper] = []
        for item in stage_one:
            title_tokens = self.base_retriever._tokenize(item.paper.title)
            abstract_tokens = self.base_retriever._tokenize(item.paper.abstract)
            keyword_tokens = self.base_retriever._tokenize(" ".join(item.paper.keywords))

            stage_two_bonus = (
                1.8 * self.base_retriever._overlap_ratio(goal_tokens, title_tokens)
                + 1.2 * self.base_retriever._overlap_ratio(goal_tokens, abstract_tokens)
                + 1.5 * self.base_retriever._overlap_ratio(goal_tokens, keyword_tokens)
            )

            signals = dict(item.signals)
            signals["rerank_bonus"] = stage_two_bonus
            rescored.append(RankedPaper(paper=item.paper, score=item.score + stage_two_bonus, signals=signals))

        rescored.sort(key=lambda x: (-x.score, x.paper.paper_id))
        return rescored[:top_k]


def load_papers_from_jsonl(path: str) -> List[PaperDocument]:
    papers: List[PaperDocument] = []
    with open(path, "r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            row = json.loads(line)
            papers.append(
                PaperDocument(
                    paper_id=str(row["paper_id"]),
                    title=str(row.get("title", "")),
                    abstract=str(row.get("abstract", "")),
                    keywords=[str(item) for item in row.get("keywords", [])],
                )
            )
    return papers
