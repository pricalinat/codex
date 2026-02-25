import unittest

from src.multi_agent.agents import (
    AgentRegistry,
    EchoAgent,
    PaperSearchAgent,
    SummaryAgent,
)
from src.multi_agent.paper_retrieval import PaperDocument


class TestAgents(unittest.TestCase):
    def test_register_and_dispatch(self):
        registry = AgentRegistry()
        registry.register(EchoAgent())

        output = registry.run("echo", {"text": "hello"})
        self.assertEqual(output["agent"], "echo")
        self.assertEqual(output["output"]["message"], "hello")

    def test_summary_agent(self):
        registry = AgentRegistry()
        registry.register(SummaryAgent())

        output = registry.run("summary", {"text": "a b c d e", "max_words": 3})
        self.assertEqual(output["output"]["summary"], "a b c")

    def test_paper_search_agent(self):
        papers = [
            PaperDocument(
                paper_id="p1",
                title="Vision Transformer for Image Recognition",
                abstract="Applying transformers to image classification tasks.",
                keywords=["vision", "image", "classification"],
            ),
            PaperDocument(
                paper_id="p2",
                title="Temporal Fusion Transformers for Time Series Forecasting",
                abstract="Forecasting with uncertainty for time series.",
                keywords=["time series", "forecasting", "uncertainty"],
            ),
        ]
        registry = AgentRegistry()
        registry.register(PaperSearchAgent(papers))

        output = registry.run(
            "paper_search",
            {
                "query": "transformer model",
                "goal": "time series forecasting with uncertainty",
                "top_k": 1,
            },
        )
        self.assertEqual(output["output"]["results"][0]["paper_id"], "p2")


if __name__ == "__main__":
    unittest.main()
