"""Bounded tool-calling agent execution loop."""

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Protocol


class AgentModel(Protocol):
    def __call__(self, prompt: str, tools: List[str]) -> dict[str, Any]: ...


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    function: Callable[..., Any]


class AgentWorkflow:
    """Execute model-selected tools with validation and a hard step limit."""

    def __init__(self, model: AgentModel, tools: List[Tool], max_steps: int = 5) -> None:
        if max_steps < 1 or len({tool.name for tool in tools}) != len(tools):
            raise ValueError("max_steps must be positive and tool names unique")
        self.model, self.tools, self.max_steps = model, {tool.name: tool for tool in tools}, max_steps

    def run(self, task: str) -> str:
        if not task.strip():
            raise ValueError("task must not be empty")
        transcript: List[str] = [f"Task: {task}"]
        for _ in range(self.max_steps):
            response = self.model("\n".join(transcript), list(self.tools))
            if response.get("final") is not None:
                return str(response["final"])
            tool_name, arguments = response.get("tool"), response.get("arguments", {})
            if tool_name not in self.tools or not isinstance(arguments, dict):
                raise ValueError("model returned an unknown tool or invalid arguments")
            result = self.tools[tool_name].function(**arguments)
            transcript.append(f"Tool {tool_name} result: {result}")
        raise RuntimeError("agent exceeded max_steps without returning a final answer")
