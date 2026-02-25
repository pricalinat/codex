from typing import Dict, List, Optional

from .agents import AgentRegistry, EchoAgent, PaperSearchAgent, SummaryAgent
from .models import EventRecord, TaskRecord
from .orchestrator import Orchestrator, PlanStep
from .paper_retrieval import PaperDocument, load_papers_from_jsonl
from .store import InMemoryStateStore


class OrchestratorService:
    def __init__(self, max_rounds: int = 100, paper_corpus_path: Optional[str] = None) -> None:
        self.store = InMemoryStateStore()
        self.orchestrator = Orchestrator(store=self.store, max_rounds=max_rounds)
        self.registry = AgentRegistry()
        self.paper_corpus_path = paper_corpus_path

    def register_builtin_agents(self) -> None:
        self.registry.register(EchoAgent())
        self.registry.register(SummaryAgent())
        self.registry.register(PaperSearchAgent(self._load_paper_corpus()))

    def submit_task(self, request: str, steps: List[PlanStep]) -> TaskRecord:
        agents: Dict[str, callable] = {}
        for name in self.registry._agents.keys():
            agents[name] = lambda payload, agent_name=name: self.registry.run(agent_name, payload)["output"]
        return self.orchestrator.run_task(request, steps, agents)

    def get_task(self, task_id: str) -> TaskRecord:
        return self.store.get_task(task_id)

    def get_events(self, task_id: str) -> List[EventRecord]:
        return self.store.get_events(task_id)

    @staticmethod
    def _default_papers() -> List[PaperDocument]:
        return [
            PaperDocument(
                paper_id="paper-ts-001",
                title="Temporal Fusion Transformers for Time Series Forecasting",
                abstract="Forecasting with uncertainty and interpretability for time series problems.",
                keywords=["time series", "forecasting", "uncertainty", "interpretability"],
            ),
            PaperDocument(
                paper_id="paper-cv-001",
                title="Vision Transformer for Image Recognition",
                abstract="Transformer-based architecture for image classification.",
                keywords=["vision", "image", "classification"],
            ),
            PaperDocument(
                paper_id="paper-nlp-001",
                title="Attention Is All You Need",
                abstract="Transformer architecture for sequence modeling in translation.",
                keywords=["transformer", "translation", "sequence"],
            ),
        ]

    def _load_paper_corpus(self) -> List[PaperDocument]:
        if self.paper_corpus_path:
            return load_papers_from_jsonl(self.paper_corpus_path)
        return self._default_papers()


def create_app(service: OrchestratorService = None):
    try:
        from fastapi import FastAPI, HTTPException
    except ImportError:
        return None

    svc = service or OrchestratorService()
    app = FastAPI(title="Multi-Agent API")

    @app.get("/health")
    def health():
        return {"ok": True}

    @app.get("/tasks/{task_id}")
    def get_task(task_id: str):
        try:
            task = svc.get_task(task_id)
        except KeyError:
            raise HTTPException(status_code=404, detail="task not found")
        return {
            "task_id": task.task_id,
            "status": task.status,
            "request": task.request,
            "result": task.result,
        }

    return app
