from typing import Any, Iterable

from agents.base import AgentResult, BaseAgent


class PipelineRunner:
    def __init__(self, agents: Iterable[BaseAgent]) -> None:
        self.agents = list(agents)

    def run(self, initial_payload: Any | None = None) -> AgentResult:
        payload = initial_payload
        result: AgentResult | None = None
        for agent in self.agents:
            result = agent.execute(payload)
            payload = result
        if result is None:
            raise ValueError("PipelineRunner requires at least one agent.")
        return result
