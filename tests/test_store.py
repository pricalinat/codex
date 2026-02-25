import unittest

from src.multi_agent.store import InMemoryStateStore


class TestStateStore(unittest.TestCase):
    def test_create_task_and_update_status(self):
        store = InMemoryStateStore()
        task = store.create_task("Generate report")

        self.assertEqual(task.status, "pending")
        self.assertEqual(store.get_task(task.task_id).request, "Generate report")

        store.update_task_status(task.task_id, "running")
        self.assertEqual(store.get_task(task.task_id).status, "running")

    def test_append_event(self):
        store = InMemoryStateStore()
        task = store.create_task("Analyze data")
        store.append_event(task.task_id, "task_started", {"step": 1})

        events = store.get_events(task.task_id)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].event_type, "task_started")
        self.assertEqual(events[0].payload["step"], 1)


if __name__ == "__main__":
    unittest.main()
