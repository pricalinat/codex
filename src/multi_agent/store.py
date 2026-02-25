from typing import Dict, List

from .models import EventRecord, TaskRecord, utc_now


class InMemoryStateStore:
    def __init__(self) -> None:
        self._tasks: Dict[str, TaskRecord] = {}
        self._events: Dict[str, List[EventRecord]] = {}

    def create_task(self, request: str) -> TaskRecord:
        task = TaskRecord(request=request)
        self._tasks[task.task_id] = task
        self._events[task.task_id] = []
        return task

    def get_task(self, task_id: str) -> TaskRecord:
        return self._tasks[task_id]

    def update_task_status(self, task_id: str, status: str) -> TaskRecord:
        task = self.get_task(task_id)
        task.status = status
        task.updated_at = utc_now()
        return task

    def append_event(self, task_id: str, event_type: str, payload: dict) -> EventRecord:
        event = EventRecord(task_id=task_id, event_type=event_type, payload=payload)
        self._events[task_id].append(event)
        return event

    def get_events(self, task_id: str) -> List[EventRecord]:
        return list(self._events[task_id])
