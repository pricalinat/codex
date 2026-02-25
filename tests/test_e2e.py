import unittest
import tempfile
from pathlib import Path

from src.multi_agent.api import OrchestratorService
from src.multi_agent.orchestrator import PlanStep


class TestE2E(unittest.TestCase):
    def test_submit_and_run_task(self):
        service = OrchestratorService(max_rounds=100)
        service.register_builtin_agents()

        steps = [
            PlanStep(step_id="s1", agent="echo", payload={"text": "hello world"}, depends_on=[]),
            PlanStep(step_id="s2", agent="summary", payload={"text": "hello world from agent", "max_words": 2}, depends_on=["s1"]),
        ]

        task = service.submit_task("compose", steps)

        self.assertEqual(task.status, "succeeded")
        self.assertEqual(task.result["s1"]["message"], "hello world")
        self.assertEqual(task.result["s2"]["summary"], "hello world")

        fetched = service.get_task(task.task_id)
        self.assertEqual(fetched.task_id, task.task_id)

        events = service.get_events(task.task_id)
        self.assertTrue(len(events) >= 2)

    def test_submit_task_with_paper_search(self):
        service = OrchestratorService(max_rounds=100)
        service.register_builtin_agents()

        steps = [
            PlanStep(
                step_id="search",
                agent="paper_search",
                payload={
                    "query": "transformer model",
                    "goal": "time series forecasting with uncertainty",
                    "top_k": 1,
                },
                depends_on=[],
            )
        ]

        task = service.submit_task("paper retrieval", steps)
        self.assertEqual(task.status, "succeeded")
        self.assertEqual(task.result["search"]["results"][0]["paper_id"], "paper-ts-001")

    def test_submit_task_with_external_paper_corpus(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            corpus_path = Path(temp_dir) / "papers.jsonl"
            corpus_path.write_text(
                '{"paper_id":"custom-1","title":"Demand Forecasting Transformer","abstract":"Uncertainty-aware time series forecasting.","keywords":["time series","forecasting","uncertainty"]}\n'
                '{"paper_id":"custom-2","title":"Vision Baseline","abstract":"Image recognition baseline.","keywords":["vision","image"]}\n',
                encoding="utf-8",
            )

            service = OrchestratorService(max_rounds=100, paper_corpus_path=str(corpus_path))
            service.register_builtin_agents()

            steps = [
                PlanStep(
                    step_id="search",
                    agent="paper_search",
                    payload={
                        "query": "transformer model",
                        "goal": "time series forecasting uncertainty",
                        "top_k": 1,
                    },
                    depends_on=[],
                )
            ]

            task = service.submit_task("paper retrieval", steps)
            self.assertEqual(task.status, "succeeded")
            self.assertEqual(task.result["search"]["results"][0]["paper_id"], "custom-1")


if __name__ == "__main__":
    unittest.main()
