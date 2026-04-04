"""LLM-backed DevOps agent runner for TTE."""

from __future__ import annotations

import copy
import json
from typing import Any, Dict, List

from openai import AsyncOpenAI

from core.config import config
from core.execution import ExecutionContext, execute_with_cache
from core.llm_cache import build_cache_key
from core.llm_utils import completion_token_limit_kwargs, run_with_rate_limit
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

BATCH_AGENT_SYSTEM_PROMPT = """\
You are coordinating multiple independent DevOps timelines.

For each branch:
1. Read the branch-specific recent history and current observation.
2. Choose exactly one tool from the allowed tool list.
3. Keep the branches independent; do not leak reasoning or state across branches.
4. Return valid JSON only.

Return this exact shape:
{
  "actions": [
    {"branch_id": "branch_0", "tool_name": "check_logs", "tool_args": {}}
  ]
}
"""


class AgentRunner:
    """Wrap the chat API so it behaves like a single-action agent."""

    def __init__(
        self,
        model: str | None = None,
        system_prompt: str | None = None,
        max_steps: int | None = None,
        execution_context: ExecutionContext | None = None,
    ):
        provider_settings = (
            config.get_provider_settings(execution_context.provider_name)
            if execution_context is not None
            else config.get_provider_settings()
        )
        self.client = AsyncOpenAI(
            api_key=provider_settings.api_key,
            base_url=provider_settings.base_url,
            timeout=getattr(config, "LLM_REQUEST_TIMEOUT_SECONDS", 20.0),
        )
        self.base_url = provider_settings.base_url
        self.model = model or provider_settings.agent_model
        self.max_steps = max_steps or config.MAX_STEPS
        self.execution_context = execution_context
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

    def _build_user_message(self, observation: str) -> Dict[str, str]:
        prompt_parts = []
        if self._pending_tool_result:
            prompt_parts.append(f"Previous tool result: {self._pending_tool_result}")
        prompt_parts.append(f"Current observation: {observation}")
        return {"role": "user", "content": "\n\n".join(prompt_parts)}

    def _serialize_history(self, max_messages: int | None = None) -> List[Dict[str, Any]]:
        history = self.conversation[1:]
        if max_messages is not None and max_messages > 0:
            history = history[-max_messages:]
        return copy.deepcopy(history)

    def snapshot(self) -> Dict[str, Any]:
        """Capture a restorable agent state snapshot."""
        return {
            "conversation": copy.deepcopy(self.conversation),
            "total_tokens": self.total_tokens,
            "pending_tool_result": self._pending_tool_result,
        }

    def restore(self, snapshot: Dict[str, Any]) -> None:
        """Restore the agent to a previously captured snapshot."""
        self.conversation = copy.deepcopy(snapshot["conversation"])
        self.total_tokens = int(snapshot["total_tokens"])
        self._pending_tool_result = snapshot["pending_tool_result"]

    @staticmethod
    def _sanitize_action(tool_name: str, tool_args: Any) -> AgentAction:
        safe_tool_name = tool_name if tool_name in ALLOWED_TOOL_NAMES else "check_logs"
        safe_tool_args = tool_args if isinstance(tool_args, dict) else {}
        return AgentAction(tool_name=safe_tool_name, tool_args=safe_tool_args)

    @classmethod
    def _payload_to_action(cls, payload: dict[str, Any]) -> AgentAction:
        action = cls._sanitize_action(payload.get("tool_name", "check_logs"), payload.get("tool_args", {}))
        action.reasoning = payload.get("reasoning")
        action.raw_llm_response = payload.get("raw_llm_response")
        return action

    @staticmethod
    def _parse_message_payload(message: Any, usage_total_tokens: int) -> dict[str, Any]:
        if getattr(message, "tool_calls", None):
            tool_call = message.tool_calls[0]
            try:
                parsed_args = json.loads(tool_call.function.arguments) if tool_call.function.arguments else {}
            except json.JSONDecodeError:
                parsed_args = {}
            return {
                "tool_name": tool_call.function.name,
                "tool_args": parsed_args,
                "reasoning": getattr(message, "content", None),
                "raw_llm_response": str(message),
                "usage_total_tokens": usage_total_tokens,
            }

        raw_content = getattr(message, "content", None) or "{}"
        try:
            payload = json.loads(raw_content)
        except json.JSONDecodeError:
            payload = {}
        return {
            "tool_name": payload.get("tool_name", "check_logs"),
            "tool_args": payload.get("tool_args", {}),
            "reasoning": payload.get("reasoning") or getattr(message, "reasoning", None),
            "raw_llm_response": raw_content,
            "usage_total_tokens": usage_total_tokens,
        }

    async def _request_action_payload(self, user_message: Dict[str, str]) -> tuple[dict[str, Any], bool]:
        if self.execution_context is None:
            response = await run_with_rate_limit(
                lambda: self.client.chat.completions.create(
                    model=self.model,
                    messages=[*self.conversation, user_message],
                    response_format={"type": "json_object"},
                    temperature=0.0,
                    **completion_token_limit_kwargs(
                        128,
                        base_url=self.base_url,
                        model=self.model,
                    ),
                )
            )
            usage_total_tokens = response.usage.total_tokens if response.usage else 0
            return self._parse_message_payload(response.choices[0].message, usage_total_tokens), False

        cache_key = build_cache_key(
            base_url=self.base_url,
            model=self.model,
            provider_profile=self.execution_context.agent_capabilities.provider_profile,
            execution_mode=self.execution_context.execution_mode,
            prompt_version=f"{self.execution_context.prompt_version}:agent_step",
            purpose="agent.step",
            payload={
                "system_prompt": self.system_prompt,
                "history": self._serialize_history(),
                "user_message": user_message,
            },
        )

        def parser(response: Any) -> dict[str, Any]:
            usage_total_tokens = response.usage.total_tokens if response.usage else 0
            return self._parse_message_payload(response.choices[0].message, usage_total_tokens)

        payload, from_cache = await execute_with_cache(
            execution_context=self.execution_context,
            cache_key=cache_key,
            request_factory=lambda: self.client.chat.completions.create(
                model=self.model,
                messages=[*self.conversation, user_message],
                response_format={"type": "json_object"},
                temperature=0.0,
                **completion_token_limit_kwargs(
                    128,
                    base_url=self.base_url,
                    model=self.model,
                ),
            ),
            parser=parser,
        )
        return payload, from_cache

    def _commit_turn(self, user_message: Dict[str, str], action: AgentAction) -> None:
        self.conversation.append(user_message)
        self.conversation.append(
            {
                "role": "assistant",
                "content": json.dumps(
                    {"tool_name": action.tool_name, "tool_args": action.tool_args},
                    sort_keys=True,
                ),
            }
        )
        self._pending_tool_result = None
        self._trim_conversation()

    def build_batch_item(self, branch_id: str, observation: str) -> dict[str, Any]:
        """Prepare one branch for the batched turbo prompt."""
        user_message = self._build_user_message(observation)
        return {
            "branch_id": branch_id,
            "history": self._serialize_history(max_messages=config.AGENT_BATCH_HISTORY * 2),
            "user_message": user_message,
        }

    def commit_batched_action(self, user_message: Dict[str, str], action: AgentAction, usage_total_tokens: int = 0) -> None:
        """Apply a batched action result to local agent state."""
        self.total_tokens += max(0, int(usage_total_tokens))
        self._commit_turn(user_message, action)

    async def step(self, observation: str) -> AgentAction:
        """Send an observation to the agent and receive a single action."""
        user_message = self._build_user_message(observation)
        try:
            payload, from_cache = await self._request_action_payload(user_message)
        except Exception as exc:
            return AgentAction(
                tool_name="check_logs",
                tool_args={},
                reasoning=f"API error fallback: {exc}",
            )

        action = self._payload_to_action(payload)
        if not from_cache:
            self.total_tokens += int(payload.get("usage_total_tokens", 0))
        self._commit_turn(user_message, action)
        return action

    def feed_tool_result(self, result_text: str) -> None:
        """Store the previous tool result for the next model step."""
        self._pending_tool_result = result_text

    def reset(self) -> None:
        """Clear conversation history for a new branch run."""
        self.conversation = [{"role": "system", "content": self.system_prompt}]
        self.total_tokens = 0
        self._pending_tool_result = None


async def run_batched_step(
    *,
    branch_items: List[dict[str, Any]],
    execution_context: ExecutionContext,
) -> dict[str, dict[str, Any]]:
    """Request one action per branch in a single provider call."""
    if not branch_items:
        return {}

    client = AsyncOpenAI(
        api_key=execution_context.api_key,
        base_url=execution_context.base_url,
        timeout=getattr(config, "LLM_REQUEST_TIMEOUT_SECONDS", 20.0),
    )
    branch_payloads = [
        {
            "branch_id": item["branch_id"],
            "recent_history": item["history"],
            "next_input": item["user_message"]["content"],
        }
        for item in branch_items
    ]
    cache_key = build_cache_key(
        base_url=execution_context.base_url,
        model=execution_context.agent_model,
        provider_profile=execution_context.agent_capabilities.provider_profile,
        execution_mode=execution_context.execution_mode,
        prompt_version=f"{execution_context.prompt_version}:batched_agent_step",
        purpose="agent.batch_step",
        payload={"branches": branch_payloads},
    )

    def parser(response: Any) -> dict[str, Any]:
        raw = response.choices[0].message.content or "{}"
        data = json.loads(raw)
        actions = data.get("actions", [])
        usage_total_tokens = response.usage.total_tokens if response.usage else 0
        per_branch_usage = usage_total_tokens // max(1, len(actions))
        payload: dict[str, Any] = {"actions": {}}
        for action in actions:
            branch_id = action.get("branch_id")
            if not branch_id:
                continue
            payload["actions"][branch_id] = {
                "tool_name": action.get("tool_name", "check_logs"),
                "tool_args": action.get("tool_args", {}),
                "reasoning": action.get("reasoning"),
                "raw_llm_response": raw,
                "usage_total_tokens": per_branch_usage,
            }
        return payload

    payload, _from_cache = await execute_with_cache(
        execution_context=execution_context,
        cache_key=cache_key,
        request_factory=lambda: client.chat.completions.create(
            model=execution_context.agent_model,
            messages=[
                {"role": "system", "content": BATCH_AGENT_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "allowed_tools": sorted(ALLOWED_TOOL_NAMES),
                            "branches": branch_payloads,
                        },
                        indent=2,
                    ),
                },
            ],
            response_format={"type": "json_object"},
            temperature=0.0,
            **completion_token_limit_kwargs(
                max(256, 96 * len(branch_items)),
                base_url=execution_context.base_url,
                model=execution_context.agent_model,
            ),
        ),
        parser=parser,
    )
    return payload.get("actions", {})
