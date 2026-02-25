from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Optional
from uuid import uuid4


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def new_id() -> str:
    return str(uuid4())


@dataclass
class TaskRecord:
    request: str
    status: str = "pending"
    task_id: str = field(default_factory=new_id)
    created_at: datetime = field(default_factory=utc_now)
    updated_at: datetime = field(default_factory=utc_now)
    result: Optional[Dict[str, Any]] = None


@dataclass
class EventRecord:
    task_id: str
    event_type: str
    payload: Dict[str, Any]
    event_id: str = field(default_factory=new_id)
    created_at: datetime = field(default_factory=utc_now)


@dataclass
class StepRecord:
    task_id: str
    step_id: str
    agent: str
    input_payload: Dict[str, Any]
    status: str = "pending"
    retries: int = 0
    max_retries: int = 0
    output_payload: Optional[Dict[str, Any]] = None
