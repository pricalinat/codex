from abc import ABC, abstractmethod
from typing import Dict, Sequence

from .paper_retrieval import PaperDocument, PaperRetriever, TwoStagePaperRetriever


class BaseAgent(ABC):
    name: str

    @abstractmethod
    def run(self, payload: dict) -> dict:
        raise NotImplementedError


class EchoAgent(BaseAgent):
    name = "echo"

    def run(self, payload: dict) -> dict:
        return {"message": payload.get("text", "")}


class SummaryAgent(BaseAgent):
    name = "summary"

    def run(self, payload: dict) -> dict:
        text = payload.get("text", "")
        max_words = int(payload.get("max_words", 20))
        words = text.split()
        return {"summary": " ".join(words[:max_words])}


class PaperSearchAgent(BaseAgent):
    name = "paper_search"

    def __init__(
        self,
        papers: Sequence[PaperDocument],
        retriever: PaperRetriever = None,
        use_two_stage: bool = True,
    ) -> None:
        self._papers = list(papers)
        if retriever is not None:
            self._retriever = retriever
        else:
            base = PaperRetriever()
            self._retriever = TwoStagePaperRetriever(base_retriever=base) if use_two_stage else base

    def run(self, payload: dict) -> dict:
        query = payload.get("query", "")
        goal = payload.get("goal", "")
        top_k = int(payload.get("top_k", 5))
        ranked = self._retriever.search(query=query, papers=self._papers, goal=goal, top_k=top_k)

        results = []
        for item in ranked:
            results.append(
                {
                    "paper_id": item.paper.paper_id,
                    "title": item.paper.title,
                    "score": item.score,
                    "signals": item.signals,
                }
            )

        return {"results": results}


class AgentRegistry:
    def __init__(self) -> None:
        self._agents: Dict[str, BaseAgent] = {}

    def register(self, agent: BaseAgent) -> None:
        self._agents[agent.name] = agent

    def run(self, agent_name: str, payload: dict) -> dict:
        agent = self._agents[agent_name]
        output = agent.run(payload)
        return {"agent": agent_name, "output": output}
