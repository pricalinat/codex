import unittest
import tempfile
from pathlib import Path

from src.multi_agent.paper_retrieval import (
    BenchmarkCase,
    PaperDocument,
    PaperRetriever,
    TwoStagePaperRetriever,
    load_papers_from_jsonl,
)


class TestPaperRetrieval(unittest.TestCase):
    def setUp(self):
        self.papers = [
            PaperDocument(
                paper_id="p1",
                title="Attention Is All You Need",
                abstract="Transformer architecture for sequence modeling in machine translation.",
                keywords=["transformer", "translation", "sequence"],
            ),
            PaperDocument(
                paper_id="p2",
                title="Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting",
                abstract="A model for time series forecasting with interpretability and uncertainty estimation.",
                keywords=["time series", "forecasting", "uncertainty", "interpretability"],
            ),
            PaperDocument(
                paper_id="p3",
                title="Vision Transformer for Image Recognition",
                abstract="Applying transformers to image classification tasks.",
                keywords=["vision", "image", "classification"],
            ),
            PaperDocument(
                paper_id="p4",
                title="Bayesian LSTM for Probabilistic Forecasting",
                abstract="Time series forecasting with Bayesian uncertainty estimates.",
                keywords=["time series", "forecasting", "uncertainty"],
            ),
        ]

    def test_goal_aligned_paper_is_ranked_first(self):
        retriever = PaperRetriever()
        ranked = retriever.search(
            query="transformer for forecasting",
            papers=self.papers,
            goal="time series forecasting with interpretability and uncertainty",
            top_k=3,
        )
        self.assertEqual(ranked[0].paper.paper_id, "p2")

    def test_iterative_tuning_reaches_usable_threshold_within_100_rounds(self):
        retriever = PaperRetriever(
            weights={
                "title_overlap": 2.0,
                "abstract_overlap": 0.1,
                "keyword_overlap": 0.1,
                "goal_overlap": 0.0,
            }
        )

        benchmark = [
            BenchmarkCase(
                query="transformer model",
                goal="time series forecasting with uncertainty and interpretability",
                relevant_ids=["p2"],
            ),
            BenchmarkCase(
                query="transformer model",
                goal="image classification",
                relevant_ids=["p3"],
            ),
        ]

        before = retriever.evaluate(benchmark, self.papers, top_k=1)
        report = retriever.tune(
            benchmark=benchmark,
            papers=self.papers,
            top_k=1,
            max_rounds=100,
            target_hit_rate=1.0,
        )
        after = retriever.evaluate(benchmark, self.papers, top_k=1)

        self.assertLessEqual(report.rounds_used, 100)
        self.assertGreater(after.hit_rate_at_k, before.hit_rate_at_k)
        self.assertGreaterEqual(after.hit_rate_at_k, 1.0)

    def test_morphology_variant_is_matched_for_relevance(self):
        papers = [
            PaperDocument(
                paper_id="distractor",
                title="Transformer forecasting baseline",
                abstract="A baseline model for forecasting.",
                keywords=["transformer", "forecasting"],
            ),
            PaperDocument(
                paper_id="target",
                title="Transformer forecasting for demand",
                abstract="An interpretable approach for multi-horizon demand forecasting.",
                keywords=["transformer", "forecasting", "interpretable"],
            ),
        ]

        retriever = PaperRetriever()
        ranked = retriever.search(
            query="transformer forecasting interpretability",
            papers=papers,
            goal="",
            top_k=1,
        )
        self.assertEqual(ranked[0].paper.paper_id, "target")

    def test_load_papers_from_jsonl(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "papers.jsonl"
            path.write_text(
                "\n".join(
                    [
                        '{"paper_id":"a1","title":"A","abstract":"AA","keywords":["x","y"]}',
                        '{"paper_id":"a2","title":"B","abstract":"BB","keywords":["z"]}',
                    ]
                ),
                encoding="utf-8",
            )
            papers = load_papers_from_jsonl(str(path))

        self.assertEqual(len(papers), 2)
        self.assertEqual(papers[0].paper_id, "a1")
        self.assertEqual(papers[1].keywords, ["z"])

    def test_tuning_reduces_hard_negative_rate(self):
        papers = [
            PaperDocument(
                paper_id="neg",
                title="Transformer model baseline",
                abstract="General architecture discussion.",
                keywords=["transformer", "model"],
            ),
            PaperDocument(
                paper_id="pos",
                title="Time series forecasting model",
                abstract="A transformer model for uncertainty-aware forecasting.",
                keywords=["time series", "forecasting", "uncertainty"],
            ),
        ]
        benchmark = [
            BenchmarkCase(
                query="transformer model",
                goal="time series forecasting with uncertainty",
                relevant_ids=["pos"],
                hard_negative_ids=["neg"],
            )
        ]
        retriever = PaperRetriever(
            weights={
                "title_overlap": 2.5,
                "abstract_overlap": 0.1,
                "keyword_overlap": 0.1,
                "goal_overlap": 0.0,
            }
        )

        before = retriever.evaluate(benchmark, papers, top_k=1)
        retriever.tune(benchmark, papers, top_k=1, max_rounds=100, target_hit_rate=1.0)
        after = retriever.evaluate(benchmark, papers, top_k=1)

        self.assertGreater(before.hard_negative_rate_at_k, after.hard_negative_rate_at_k)
        self.assertEqual(after.hard_negative_rate_at_k, 0.0)

    def test_two_stage_reranker_promotes_goal_aligned_paper(self):
        papers = [
            PaperDocument(
                paper_id="d1",
                title="Transformer model and architecture",
                abstract="Core concepts of transformer model design.",
                keywords=["transformer", "architecture"],
            ),
            PaperDocument(
                paper_id="t1",
                title="Transformer model for operations",
                abstract="Forecasting demand uncertainty in time series settings.",
                keywords=["time series", "forecasting", "uncertainty"],
            ),
        ]
        retriever = TwoStagePaperRetriever(base_retriever=PaperRetriever())
        ranked = retriever.search(
            query="transformer model",
            papers=papers,
            goal="time series forecasting uncertainty",
            top_k=1,
        )
        self.assertEqual(ranked[0].paper.paper_id, "t1")

    def test_deterministic_tiebreaker_ranks_by_paper_id(self):
        papers = [
            PaperDocument(
                paper_id="aaa",
                title="Deep Learning for NLP",
                abstract="A paper about deep learning.",
                keywords=["deep learning", "nlp"],
            ),
            PaperDocument(
                paper_id="bbb",
                title="Deep Learning for NLP",
                abstract="A paper about deep learning.",
                keywords=["deep learning", "nlp"],
            ),
            PaperDocument(
                paper_id="ccc",
                title="Deep Learning for NLP",
                abstract="A paper about deep learning.",
                keywords=["deep learning", "nlp"],
            ),
        ]
        retriever = PaperRetriever()
        ranked = retriever.search(
            query="deep learning",
            papers=papers,
            goal="",
            top_k=3,
        )
        self.assertEqual(ranked[0].paper.paper_id, "aaa")
        self.assertEqual(ranked[1].paper.paper_id, "bbb")
        self.assertEqual(ranked[2].paper.paper_id, "ccc")

    def test_deterministic_ranking_consistent_across_calls(self):
        papers = [
            PaperDocument(
                paper_id="x1",
                title="Neural Networks Overview",
                abstract="Introduction to neural networks.",
                keywords=["neural", "networks"],
            ),
            PaperDocument(
                paper_id="a1",
                title="Neural Networks Overview",
                abstract="Introduction to neural networks.",
                keywords=["neural", "networks"],
            ),
        ]
        retriever = PaperRetriever()
        results = []
        for _ in range(5):
            ranked = retriever.search(
                query="neural networks",
                papers=papers,
                goal="",
                top_k=2,
            )
            results.append([r.paper.paper_id for r in ranked])
        for result in results:
            self.assertEqual(result, results[0])

    def test_unknown_abbreviation_expanded_from_document_context(self):
        papers = [
            PaperDocument(
                paper_id="p1",
                title="Hierarchical Temporal Memory (HTM) for Time Series",
                abstract="A novel approach to time series prediction using HTM.",
                keywords=["temporal", "memory", "prediction"],
            ),
            PaperDocument(
                paper_id="p2",
                title="Standard time series methods",
                abstract="Traditional approaches to forecasting.",
                keywords=["forecasting", "time series"],
            ),
        ]
        retriever = PaperRetriever()
        ranked = retriever.search(
            query="HTM",
            papers=papers,
            goal="",
            top_k=2,
        )
        self.assertEqual(ranked[0].paper.paper_id, "p1")

    def test_abbreviation_expansion_invoked_from_search(self):
        """Regression test: search() must invoke unknown abbreviation expansion path."""
        papers = [
            PaperDocument(
                paper_id="target",
                title="Hierarchical Temporal Memory for Time Series",
                abstract="A novel approach to time series prediction.",
                keywords=["temporal", "memory", "prediction"],
            ),
            PaperDocument(
                paper_id="other",
                title="Standard forecasting methods",
                abstract="Traditional approaches to forecasting.",
                keywords=["forecasting", "time series"],
            ),
        ]

        expansion_call_count = [0]

        class SpyPaperRetriever(PaperRetriever):
            def _expand_unknown_abbreviation(self, token: str, papers):
                expansion_call_count[0] += 1
                if token == "HTM":
                    return ["hierarchical", "temporal", "memory"]
                return []

        retriever = SpyPaperRetriever()
        ranked = retriever.search(
            query="HTM",
            papers=papers,
            goal="",
            top_k=2,
        )

        self.assertGreater(
            expansion_call_count[0],
            0,
            "_expand_unknown_abbreviation was NOT invoked from search()",
        )
        self.assertEqual(ranked[0].paper.paper_id, "target")

    def test_add_synonym_runtime_extends_abbreviation_handling(self):
        papers = [
            PaperDocument(
                paper_id="target",
                title="CRF for sequence labeling",
                abstract="Conditional Random Fields for labeling tasks.",
                keywords=["crf", "labeling"],
            ),
            PaperDocument(
                paper_id="other",
                title="General sequence methods",
                abstract="Overview of sequence processing.",
                keywords=["sequence", "processing"],
            ),
        ]
        retriever = PaperRetriever()
        retriever.search(query="CRF", papers=papers, goal="", top_k=1)
        retriever.add_synonym("crf", ["conditional", "random", "fields"])
        ranked_after = retriever.search(
            query="CRF",
            papers=papers,
            goal="",
            top_k=1,
        )
        self.assertEqual(ranked_after[0].paper.paper_id, "target")

    def test_abbreviation_detection_requires_uppercase(self):
        retriever = PaperRetriever()
        self.assertTrue(retriever._is_abbreviation("NLP"))
        self.assertTrue(retriever._is_abbreviation("CRF"))
        self.assertTrue(retriever._is_abbreviation("HTM"))
        self.assertFalse(retriever._is_abbreviation("nlp"))
        self.assertFalse(retriever._is_abbreviation("Natural"))
        self.assertFalse(retriever._is_abbreviation("A"))
        self.assertFalse(retriever._is_abbreviation("LONGWORD"))

    def test_phrase_order_match_is_prioritized(self):
        papers = [
            PaperDocument(
                paper_id="neg",
                title="Forecasting time series with transformer",
                abstract="Same tokens in a weaker order.",
                keywords=["forecasting", "time", "series"],
            ),
            PaperDocument(
                paper_id="pos",
                title="Time series forecasting with transformer",
                abstract="Exact phrase alignment with query intent.",
                keywords=["time series", "forecasting"],
            ),
        ]
        retriever = PaperRetriever()
        ranked = retriever.search(
            query="time series forecasting",
            papers=papers,
            goal="",
            top_k=1,
        )
        self.assertEqual(ranked[0].paper.paper_id, "pos")

    def test_two_stage_tiebreaker_matches_stage_one_ordering(self):
        papers = [
            PaperDocument(
                paper_id="aaa",
                title="Deep Learning Overview",
                abstract="Introduction to deep learning.",
                keywords=["deep", "learning"],
            ),
            PaperDocument(
                paper_id="bbb",
                title="Deep Learning Overview",
                abstract="Introduction to deep learning.",
                keywords=["deep", "learning"],
            ),
            PaperDocument(
                paper_id="ccc",
                title="Deep Learning Overview",
                abstract="Introduction to deep learning.",
                keywords=["deep", "learning"],
            ),
        ]

        retriever = TwoStagePaperRetriever(base_retriever=PaperRetriever())
        ranked = retriever.search(
            query="deep learning",
            papers=papers,
            goal="machine learning",
            top_k=3,
        )

        self.assertEqual(ranked[0].paper.paper_id, "aaa")
        self.assertEqual(ranked[1].paper.paper_id, "bbb")
        self.assertEqual(ranked[2].paper.paper_id, "ccc")

    def test_lowercase_htm_does_not_invoke_expand_unknown_abbreviation(self):
        """Lowercase 'htm' must not invoke _expand_unknown_abbreviation from search."""
        papers = [
            PaperDocument(
                paper_id="p1",
                title="Hierarchical Temporal Memory (HTM) for Time Series",
                abstract="A novel approach to time series prediction using HTM.",
                keywords=["temporal", "memory", "prediction"],
            ),
        ]

        expansion_call_count = [0]

        class SpyPaperRetriever(PaperRetriever):
            def _expand_unknown_abbreviation(self, token: str, papers):
                expansion_call_count[0] += 1
                return super()._expand_unknown_abbreviation(token, papers)

        retriever = SpyPaperRetriever()
        ranked = retriever.search(
            query="htm",
            papers=papers,
            goal="",
            top_k=1,
        )

        self.assertEqual(
            expansion_call_count[0],
            0,
            "_expand_unknown_abbreviation was INVOKED for lowercase 'htm' but should NOT be",
        )

    def test_uppercase_htm_invokes_expand_unknown_abbreviation(self):
        """Uppercase 'HTM' must invoke _expand_unknown_abbreviation from search."""
        papers = [
            PaperDocument(
                paper_id="p1",
                title="Hierarchical Temporal Memory (HTM) for Time Series",
                abstract="A novel approach to time series prediction using HTM.",
                keywords=["temporal", "memory", "prediction"],
            ),
        ]

        expansion_call_count = [0]

        class SpyPaperRetriever(PaperRetriever):
            def _expand_unknown_abbreviation(self, token: str, papers):
                expansion_call_count[0] += 1
                if token == "HTM":
                    return ["hierarchical", "temporal", "memory"]
                return super()._expand_unknown_abbreviation(token, papers)

        retriever = SpyPaperRetriever()
        ranked = retriever.search(
            query="HTM",
            papers=papers,
            goal="",
            top_k=1,
        )

        self.assertGreater(
            expansion_call_count[0],
            0,
            "_expand_unknown_abbreviation was NOT invoked for uppercase 'HTM' but should be",
        )
        self.assertEqual(ranked[0].paper.paper_id, "p1")



    def test_expansion_with_lowercase_words(self):
        """Expansion phrases with lowercase words should be extracted (e.g., 'long short-term memory')."""
        papers = [
            PaperDocument(
                paper_id="p1",
                title="Long Short-Term Memory (LSTM) Networks",
                abstract="LSTM is a type of recurrent neural network.",
                keywords=["lstm", "rnn"],
            ),
        ]
        retriever = PaperRetriever()
        expansions = retriever._expand_unknown_abbreviation("LSTM", papers)
        # Uses the same tokenizer path as retrieval: "short-term" -> "short", "term"
        self.assertIn("long", expansions)
        self.assertIn("short", expansions)
        self.assertIn("term", expansions)
        self.assertIn("memory", expansions)

    def test_expansion_with_hyphenated_words(self):
        """Expansion phrases with hyphenated words should be extracted."""
        papers = [
            PaperDocument(
                paper_id="p1",
                title="End-to-End (E2E) Learning",
                abstract="E2E training simplifies the pipeline.",
                keywords=["e2e", "learning"],
            ),
        ]
        retriever = PaperRetriever()
        expansions = retriever._expand_unknown_abbreviation("E2E", papers)
        self.assertIn("end", expansions)
        self.assertIn("to", expansions)

    def test_expansion_token_selection_deterministic(self):
        """Expansion token selection should be deterministic (same order every time)."""
        papers = [
            PaperDocument(
                paper_id="p1",
                title="Deep Learning (DL) Methods",
                abstract="DL techniques for various tasks.",
                keywords=["deep learning"],
            ),
            PaperDocument(
                paper_id="p2",
                title="Deep Learning (DL) Approaches",
                abstract="Alternative DL approaches.",
                keywords=["deep"],
            ),
        ]
        retriever = PaperRetriever()
        # Run multiple times and check order is consistent
        results = []
        for _ in range(10):
            expansions = retriever._expand_unknown_abbreviation("DL", papers)
            results.append(tuple(expansions))
        # All results should be identical
        self.assertEqual(len(set(results)), 1, "Expansion token selection is non-deterministic")

if __name__ == "__main__":
    unittest.main()
