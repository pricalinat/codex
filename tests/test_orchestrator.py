import unittest

from src.multi_agent.orchestrator import Orchestrator, PlanStep
from src.multi_agent.store import InMemoryStateStore


class TestOrchestrator(unittest.TestCase):
    def test_executes_steps_successfully(self):
        store = InMemoryStateStore()
        orchestrator = Orchestrator(store=store, max_rounds=100)

        steps = [
            PlanStep(step_id="s1", agent="echo", payload={"text": "hello"}, depends_on=[]),
            PlanStep(step_id="s2", agent="echo", payload={"text": "world"}, depends_on=["s1"]),
        ]

        agents = {
            "echo": lambda payload: {"message": payload["text"]},
        }

        task = orchestrator.run_task("test", steps, agents)
        self.assertEqual(task.status, "succeeded")
        self.assertEqual(task.result["s2"]["message"], "world")

    def test_retries_failed_step_until_success(self):
        store = InMemoryStateStore()
        orchestrator = Orchestrator(store=store, max_rounds=100)

        attempts = {"count": 0}

        def flaky(_payload):
            attempts["count"] += 1
            if attempts["count"] < 3:
                raise RuntimeError("temporary")
            return {"ok": True}

        steps = [
            PlanStep(step_id="s1", agent="flaky", payload={}, depends_on=[], max_retries=3),
        ]

        task = orchestrator.run_task("retry test", steps, {"flaky": flaky})
        self.assertEqual(task.status, "succeeded")
        self.assertEqual(attempts["count"], 3)

    def test_respects_max_rounds(self):
        store = InMemoryStateStore()
        orchestrator = Orchestrator(store=store, max_rounds=1)

        steps = [
            PlanStep(step_id="s1", agent="echo", payload={"text": "hello"}, depends_on=["missing"]),
        ]

        task = orchestrator.run_task("round limit", steps, {"echo": lambda p: p})
        self.assertEqual(task.status, "failed")


if __name__ == "__main__":
    unittest.main()
