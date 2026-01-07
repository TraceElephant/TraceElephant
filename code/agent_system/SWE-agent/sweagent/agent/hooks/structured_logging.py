"""Structured logging hook for SWE-agent."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from sweagent.agent.hooks.abstract import AbstractAgentHook
from sweagent.agent.problem_statement import ProblemStatement
from sweagent.types import AgentInfo, Trajectory


class StructuredLoggingHook(AbstractAgentHook):
    """Hook to generate structured logs with detailed LLM step tracking."""

    def __init__(self, output_dir: Path, problem_statement: ProblemStatement):
        self.output_dir = output_dir
        self.problem_statement = problem_statement
        self.llm_steps_dir = output_dir / "llm_steps"
        self.images_dir = self.llm_steps_dir / "images"
        self.step_counter = 0
        self._current_request: dict[str, Any] | None = None
        self._initial_query_messages: list[dict] = []

        # Create directory structure
        self.llm_steps_dir.mkdir(parents=True, exist_ok=True)
        self.images_dir.mkdir(exist_ok=True)

    def on_model_query(self, *, messages: list[dict[str, str]], agent: str):
        """Record LLM request start"""
        self.step_counter += 1
        self._current_request = {
            "messages": messages,
            "agent": agent,
            "step_number": self.step_counter,
        }

        # Save initial query for generating summary
        if self.step_counter == 1:
            self._initial_query_messages = messages

    def on_model_query_complete(
        self,
        *,
        messages: list[dict[str, str]],
        response: dict,
        model_name: str,
        agent: str,
    ):
        """Record LLM response and generate step_*.json"""
        step_file = self.llm_steps_dir / f"step_{self.step_counter}.json"

        step_data = {
            "request": {
                "messages": self._sanitize_messages(messages),
                "model": model_name,
                "stream": False,
            },
            "response": self._serialize_response(response, model_name),
        }

        step_file.write_text(json.dumps(step_data, indent=2, ensure_ascii=False))

        # Reserved: Extract images (if messages contain multimodal content)
        # self._extract_images(messages, self.step_counter)

    def on_run_done(self, *, trajectory: Trajectory, info: AgentInfo):
        """Generate summary.json and judge.json when run completes"""
        self._generate_summary(trajectory, info)
        self._generate_judge(info)

    def _sanitize_messages(self, messages: list[dict]) -> list[dict]:
        """Clean message content, remove non-serializable objects"""
        sanitized = []
        for msg in messages:
            clean_msg = {
                "role": msg.get("role", "unknown"),
                "content": self._sanitize_content(msg.get("content", "")),
            }
            # Keep other serializable fields
            for key in ["tool_calls", "tool_call_ids", "thinking_blocks"]:
                if key in msg and msg[key] is not None:
                    clean_msg[key] = msg[key]
            sanitized.append(clean_msg)
        return sanitized

    def _sanitize_content(self, content: Any) -> str | list:
        """Clean content field"""
        if isinstance(content, str):
            return content
        elif isinstance(content, list):
            # Multimodal content: text + images
            return content
        else:
            return str(content)

    def _serialize_response(self, response: dict, model_name: str) -> str:
        """Serialize LLM response as ChatCompletion string representation"""
        # Build a string representation similar to ChatCompletion object
        message_content = response.get("message", "")
        tool_calls = response.get("tool_calls")
        thinking_blocks = response.get("thinking_blocks")

        # Build message representation
        message_parts = [
            f"content={repr(message_content)}",
            "refusal=None",
            "role='assistant'",
            "annotations=[]",
            "audio=None",
            "function_call=None",
        ]

        if tool_calls:
            message_parts.append(f"tool_calls={tool_calls}")
        else:
            message_parts.append("tool_calls=None")

        message_repr = f"ChatCompletionMessage({', '.join(message_parts)})"

        # Build choice representation
        finish_reason = "tool_calls" if tool_calls else "stop"
        choice_repr = (
            f"Choice(finish_reason='{finish_reason}', index=0, logprobs=None, "
            f"message={message_repr})"
        )

        # Build full ChatCompletion representation
        completion_repr = (
            f"ChatCompletion("
            f"id='chatcmpl-step-{self.step_counter}', "
            f"choices=[{choice_repr}], "
            f"created={int(time.time())}, "
            f"model='{model_name}', "
            f"object='chat.completion', "
            f"service_tier=None, "
            f"system_fingerprint=None, "
            f"usage=CompletionUsage(completion_tokens=0, prompt_tokens=0, total_tokens=0, "
            f"completion_tokens_details=None, prompt_tokens_details=None)"
            f")"
        )

        return completion_repr

    def _extract_images(self, messages: list[dict], step_number: int):
        """Extract and save images (reserved interface, not implemented yet)"""
        # TODO: Implement image extraction from multimodal content
        # Iterate through messages, find type="image_url" or type="image" content
        # Save to images/step_{step_number}_image_{index}.png
        pass

    def _generate_summary(self, trajectory: Trajectory, info: AgentInfo):
        """Generate summary.json"""
        summary = {
            "history": self._build_history(trajectory),
            "question": self._extract_question(),
            "question_ID": self.problem_statement.id,
            "is_corrected": False,
            "exit_status": info.get("exit_status", ""),
            "submission": info.get("submission", ""),
            "model_stats": info.get("model_stats", {}),
        }

        summary_file = self.output_dir / "summary.json"
        summary_file.write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    def _build_history(self, trajectory: Trajectory) -> list[dict]:
        """Build concise history showing main trajectory flow"""
        history = []

        for idx, step in enumerate(trajectory, 1):
            # Add agent's thought if present
            if step.get("thought"):
                history.append({
                    "content": step["thought"],
                    "role": "agent (thought)",
                })

            # Add action
            if step.get("action"):
                history.append({
                    "content": step["action"],
                    "role": "agent (action)",
                })

            # Add observation from environment
            if step.get("observation"):
                history.append({
                    "content": step["observation"],
                    "role": "environment",
                })

        return history

    def _extract_question(self) -> str:
        """Extract original question description"""
        # Get from problem_statement
        try:
            return self.problem_statement.get_problem_statement()
        except Exception:
            return ""

    def _generate_judge(self, info: AgentInfo):
        """Generate judge.json"""
        exit_status = info.get("exit_status", "")
        submission = info.get("submission", "")

        # Determine if successful
        is_correct = self._evaluate_success(exit_status, submission)
        reason = self._get_judge_reason(exit_status, submission, is_correct)

        judge = {
            "extracted_answer": submission or "",
            "is_correct": is_correct,
            "reason": reason,
        }

        judge_file = self.output_dir / "judge.json"
        judge_file.write_text(json.dumps(judge, indent=2, ensure_ascii=False))

    def _evaluate_success(self, exit_status: str | None, submission: str | None) -> bool:
        """Evaluate if task was successful"""
        if not exit_status:
            return False

        exit_status_str = str(exit_status).lower()

        # Success criteria: exit_status contains "submitted"
        if "submitted" in exit_status_str:
            return True

        # Failure criteria: contains "exit_" prefix (exit_cost, exit_error, etc.)
        if "exit_" in exit_status_str:
            return False

        # Other cases: check if there's a valid submission
        return bool(submission and submission.strip())

    def _get_judge_reason(
        self, exit_status: str | None, submission: str | None, is_correct: bool
    ) -> str:
        """Generate judgment reason"""
        if is_correct:
            return f"Task completed successfully. Exit status: {exit_status or 'unknown'}"
        else:
            if not submission:
                return f"No valid submission. Exit status: {exit_status or 'unknown'}"
            return f"Task failed or incomplete. Exit status: {exit_status or 'unknown'}"
