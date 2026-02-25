from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

from .models import TaskRecord
from .store import InMemoryStateStore


@dataclass
class PlanStep:
    step_id: str
    agent: str
    payload: dict
    depends_on: List[str]
    max_retries: int = 0


class Orchestrator:
    def __init__(self, store: InMemoryStateStore, max_rounds: int = 100) -> None:
        self.store = store
        self.max_rounds = max_rounds

    def run_task(
        self,
        request: str,
        steps: List[PlanStep],
        agents: Dict[str, Callable[[dict], dict]],
    ) -> TaskRecord:
        task = self.store.create_task(request)
        self.store.update_task_status(task.task_id, "running")

        status: Dict[str, str] = {step.step_id: "pending" for step in steps}
        retries: Dict[str, int] = {step.step_id: 0 for step in steps}
        result: Dict[str, dict] = {}
        round_count = 0

        step_index = {step.step_id: step for step in steps}

        while round_count < self.max_rounds:
            round_count += 1
            progress_made = False

            for step in steps:
                if status[step.step_id] == "succeeded":
                    continue
                if any(status.get(dep) != "succeeded" for dep in step.depends_on):
                    continue

                progress_made = True
                status[step.step_id] = "running"
                self.store.append_event(task.task_id, "step_started", {"step_id": step.step_id})

                try:
                    agent = agents[step.agent]
                    output = agent(step.payload)
                    result[step.step_id] = output
                    status[step.step_id] = "succeeded"
                    self.store.append_event(
                        task.task_id,
                        "step_succeeded",
                        {"step_id": step.step_id, "output": output},
                    )
                except Exception as exc:  # pragma: no cover - branch validated by behavior
                    retries[step.step_id] += 1
                    self.store.append_event(
                        task.task_id,
                        "step_failed",
                        {
                            "step_id": step.step_id,
                            "error": str(exc),
                            "retry": retries[step.step_id],
                        },
                    )
                    if retries[step.step_id] > step_index[step.step_id].max_retries:
                        status[step.step_id] = "failed"
                        self.store.update_task_status(task.task_id, "failed")
                        task.result = result
                        return task
                    status[step.step_id] = "pending"

            if all(value == "succeeded" for value in status.values()):
                self.store.update_task_status(task.task_id, "succeeded")
                task.result = result
                return task

            if not progress_made:
                self.store.append_event(task.task_id, "orchestrator_stalled", {"round": round_count})

        self.store.update_task_status(task.task_id, "failed")
        self.store.append_event(task.task_id, "orchestrator_max_rounds", {"rounds": self.max_rounds})
        task.result = result
        return task
