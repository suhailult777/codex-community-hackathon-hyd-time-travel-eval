"""LLM-backed DevOps agent runner for TTE."""

from __future__ import annotations

import json
from typing import Any, Dict, List

from openai import AsyncOpenAI

from core.config import config
from core.llm_utils import run_with_rate_limit
from core.models import AgentAction

DEVOPS_TOOLS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "run_tests",
            "description": "Run the test suite and check if all tests pass.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "deploy",
            "description": "Deploy the current code to production.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "rollback",
            "description": "Rollback to the previous stable deployment version.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "restart_service",
            "description": "Restart the application service to restore it to a healthy state.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_logs",
            "description": "Read the most recent application and system logs.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "scale_resources",
            "description": "Scale up compute resources to reduce utilization.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "retry_connection",
            "description": "Retry the connection to a downstream dependency.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "finish_task",
            "description": "Declare the task complete once the system is stable.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]

ALLOWED_TOOL_NAMES = {tool["function"]["name"] for tool in DEVOPS_TOOLS}

AGENT_SYSTEM_PROMPT = """\
You are a DevOps AI agent responsible for deploying and maintaining a production system.

You receive observations about the current system state at each step. Your goal is to
successfully deploy the code to production and ensure the system is healthy afterwards.

Rules:
1. Respond with exactly one tool decision.
2. Before deploying, check that preconditions are met.
3. If deployment fails, diagnose the issue and either fix it or rollback.
4. Call finish_task only when deployment is complete and the system is stable.
5. You have a maximum of {max_steps} steps.

Return only valid JSON with this exact shape:
{{"tool_name": "<one of: {allowed_tools}>", "tool_args": {{}}}}
"""


class AgentRunner:
    """Wrap the chat API so it behaves like a single-action agent."""

    def __init__(
        self,
        model: str | None = None,
        system_prompt: str | None = None,
        max_steps: int | None = None,
    ):
        self.client = AsyncOpenAI(
            api_key=config.API_KEY,
            base_url=config.API_BASE_URL,
            timeout=getattr(config, "LLM_REQUEST_TIMEOUT_SECONDS", 20.0),
        )
        self.model = model or config.MODEL_AGENT
        self.max_steps = max_steps or config.MAX_STEPS
        self.system_prompt = (system_prompt or AGENT_SYSTEM_PROMPT).format(
            max_steps=self.max_steps,
            allowed_tools=", ".join(sorted(ALLOWED_TOOL_NAMES)),
        )
        self.conversation: List[Dict[str, Any]] = [{"role": "system", "content": self.system_prompt}]
        self.total_tokens = 0
        self._pending_tool_result: str | None = None

    def _trim_conversation(self, max_messages: int = 8) -> None:
        """Keep a small rolling history to reduce prompt size and latency."""
        if len(self.conversation) <= max_messages + 1:
            return
        self.conversation = [self.conversation[0], *self.conversation[-max_messages:]]

    async def _request_completion(self):
        """Run the model call with shared pacing and retries."""
        return await run_with_rate_limit(
            lambda: self.client.chat.completions.create(
                model=self.model,
                messages=self.conversation,
                response_format={"type": "json_object"},
                temperature=0.0,
                max_tokens=128,
            )
        )

    @staticmethod
    def _sanitize_action(tool_name: str, tool_args: Any) -> AgentAction:
        safe_tool_name = tool_name if tool_name in ALLOWED_TOOL_NAMES else "check_logs"
        safe_tool_args = tool_args if isinstance(tool_args, dict) else {}
        return AgentAction(tool_name=safe_tool_name, tool_args=safe_tool_args)

    async def step(self, observation: str) -> AgentAction:
        """Send an observation to the agent and receive a single action."""
        prompt_parts = []
        if self._pending_tool_result:
            prompt_parts.append(f"Previous tool result: {self._pending_tool_result}")
            self._pending_tool_result = None
        prompt_parts.append(f"Current observation: {observation}")
        self.conversation.append({"role": "user", "content": "\n\n".join(prompt_parts)})
        self._trim_conversation()

        try:
            response = await self._request_completion()
        except Exception as exc:
            return AgentAction(
                tool_name="check_logs",
                tool_args={},
                reasoning=f"API error fallback: {exc}",
            )

        if response.usage:
            self.total_tokens += response.usage.total_tokens

        message = response.choices[0].message

        if message.tool_calls:
            tool_call = message.tool_calls[0]
            try:
                parsed_args = json.loads(tool_call.function.arguments) if tool_call.function.arguments else {}
            except json.JSONDecodeError:
                parsed_args = {}

            action = self._sanitize_action(tool_call.function.name, parsed_args)
            action.reasoning = message.content
            action.raw_llm_response = str(message)
            self.conversation.append(message.model_dump())
            self._trim_conversation()
            return action

        if message.content:
            try:
                payload = json.loads(message.content)
            except json.JSONDecodeError:
                payload = {}

            action = self._sanitize_action(payload.get("tool_name", "check_logs"), payload.get("tool_args", {}))
            action.reasoning = getattr(message, "reasoning", None)
            action.raw_llm_response = str(message)
            self.conversation.append(message.model_dump())
            self._trim_conversation()
            return action

        return AgentAction(
            tool_name="check_logs",
            tool_args={},
            reasoning="No usable action returned - fallback.",
        )

    def feed_tool_result(self, result_text: str) -> None:
        """Store the previous tool result for the next model step."""
        self._pending_tool_result = result_text

    def reset(self) -> None:
        """Clear conversation history for a new branch run."""
        self.conversation = [{"role": "system", "content": self.system_prompt}]
        self.total_tokens = 0
        self._pending_tool_result = None
