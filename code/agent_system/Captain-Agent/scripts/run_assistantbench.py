#!/usr/bin/env python
import argparse
import json
import os
import re
import glob
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from dotenv import load_dotenv

import autogen
from autogen.agentchat.contrib.meta_agent import MetaAgent
from autogen.agentchat.contrib.meta_user_proxy_agent import MetaUserProxyAgent


ASSISTANTBENCH_SYSTEM_TEMPLATE = """# Task
You need to solve the below question given by a user. This task may require web search, data analysis, and multi-step reasoning.

# Question
{question}

# Important Constraint
You MUST solve this problem within {max_round} rounds of conversation. Plan your approach efficiently and focus on the most direct path to the answer.

# Output format (MANDATORY)
Please respond in the following structure:
## ANSWER
[concise final answer]
## REASON
[brief reasoning or evidence used to reach the answer]

If you obtain and output the final answer, please output 'terminate' in **uppercase** format.
"""

DEFAULT_BUILDING_TASK = """- Web search specialist skilled at finding and analyzing online information
- Python coder who can process data and perform calculations
- Checker that verifies answers and enforces grounding in cited evidence"""


def _load_task(dataset_path: Path, task_id: Optional[str], task_index: Optional[int]) -> Optional[Dict]:
    with dataset_path.open("r") as fh:
        records = [json.loads(line) for line in fh]
    if task_id:
        for record in records:
            if record.get("id") == task_id:
                return record
    if task_index is not None:
        if 0 <= task_index < len(records):
            return records[task_index]
    return None


def _build_llm_configs(args, repo_root: Path):
    filter_dict = {"model": [args.model]} if args.model else None
    config_list = autogen.config_list_from_json(args.config_list, filter_dict=filter_dict)
    general_llm_config = {"temperature": 0.1, "top_p": 0.95, "timeout": 60, "config_list": config_list}
    tool_root = repo_root / "tools"
    nested_mode_config = {
        "autobuild_init_config": {
            "config_file_or_env": args.config_list,
            "builder_model": [args.model],
            "agent_model": [args.model],
        },
        "autobuild_build_config": {
            "default_llm_config": {
                "temperature": 0.1,
                "top_p": 0.95,
                "max_tokens": 4096,
                "cache_seed": None,
                "timeout": 60,
            },
            "code_execution_config": {
                "timeout": 300,
                "work_dir": str(args.work_dir),
                "last_n_messages": 2,
                "use_docker": False,
            },
            "coding": True,
            "library_path_or_json": str(repo_root / "agent_library.json"),
        },
        "autobuild_tool_config": {
            "tool_corpus": str(tool_root / "tool_description.tsv"),
            "tool_root": str(tool_root),
            "retriever": "all-MiniLM-L6-v2",
        },
        "group_chat_config": {"max_round": args.max_round},
        "group_chat_llm_config": general_llm_config.copy(),
    }
    return general_llm_config, nested_mode_config


def _run_single_task(
    meta_agent: MetaAgent,
    meta_user_proxy: MetaUserProxyAgent,
    question: str,
    today: str,
    max_round: int = 20,
) -> tuple[str, List[Dict], Dict]:
    prompt = ASSISTANTBENCH_SYSTEM_TEMPLATE.format(today=today, question=question.strip(), max_round=max_round)
    meta_user_proxy.initiate_chat(meta_agent, silent=False, message=prompt)

    # Collect conversation history and system prompts
    conversation_history = []
    system_prompts = {}

    # Get the nested group chat history and agent information
    try:
        # Access the cached agent_list and groupchat from meta_user_proxy
        if hasattr(meta_user_proxy, '_agent_list_cache') and meta_user_proxy._agent_list_cache:
            for group_name, agent_list in meta_user_proxy._agent_list_cache.items():
                # Collect system prompts from each agent
                for agent in agent_list:
                    if hasattr(agent, 'name') and hasattr(agent, 'system_message'):
                        system_prompts[agent.name] = agent.system_message

                # Try to get messages directly from the GroupChat object (preferred method)
                if hasattr(meta_user_proxy, '_groupchat_cache') and group_name in meta_user_proxy._groupchat_cache:
                    groupchat = meta_user_proxy._groupchat_cache[group_name]
                    if hasattr(groupchat, 'messages'):
                        conversation_history = groupchat.messages
                        print(f"[INFO] Retrieved {len(conversation_history)} messages from GroupChat.messages", flush=True)

                # Fallback: Get the group chat messages from the first agent's chat_messages
                if not conversation_history and agent_list and hasattr(agent_list[0], 'chat_messages'):
                    for key, messages in agent_list[0].chat_messages.items():
                        conversation_history = messages
                        print(f"[INFO] Retrieved {len(messages)} messages from agent.chat_messages (fallback)", flush=True)
                        break

                # If we found the conversation history, stop searching
                if conversation_history:
                    break

        # Fallback: if we still don't have conversation history, try to get from meta_user_proxy
        if not conversation_history:
            if hasattr(meta_user_proxy, 'chat_messages') and meta_user_proxy.chat_messages:
                for key, messages in meta_user_proxy.chat_messages.items():
                    conversation_history = messages
                    print(f"[INFO] Retrieved {len(messages)} messages from meta_user_proxy (final fallback)", flush=True)
                    break
    except Exception as e:
        print(f"Warning: Could not extract conversation details: {e}", flush=True)
        import traceback
        traceback.print_exc()

    # Debug: print first few messages to understand structure
    if conversation_history:
        print(f"\n[DEBUG] Inspecting first 3 messages:", flush=True)
        for i, msg in enumerate(conversation_history[:3]):
            print(f"  Message {i}: keys={list(msg.keys())}, role={msg.get('role')}, name={msg.get('name')}", flush=True)

    # Clean and normalize conversation history
    cleaned_history = []
    for msg in conversation_history:
        if not isinstance(msg, dict):
            continue

        # Create a cleaned message with only necessary fields
        cleaned_msg = {}

        # Copy content (role field is removed for cleaner logs)
        if 'content' in msg:
            cleaned_msg['content'] = msg['content']

        # Ensure every message has a name field
        # Check if 'name' exists and is not empty
        if 'name' in msg and msg['name']:
            cleaned_msg['name'] = msg['name']
        else:
            # If no name or empty name, try to infer from the message structure
            # In group chat, the name should always be present, but as fallback:
            if msg.get('role') == 'assistant':
                # For assistant messages without name, keep looking for hints
                cleaned_msg['name'] = 'assistant'
            elif msg.get('role') == 'user':
                cleaned_msg['name'] = 'user'
            elif msg.get('role') == 'system':
                cleaned_msg['name'] = 'system'
            elif msg.get('role') == 'tool':
                cleaned_msg['name'] = 'tool'
            else:
                cleaned_msg['name'] = 'unknown'

        # Skip messages without content
        if 'content' not in cleaned_msg or not cleaned_msg['content']:
            continue

        cleaned_history.append(cleaned_msg)

    return "", cleaned_history, system_prompts


def main():
    # Load environment variables from .env if present (SERPER_API_KEY, etc.)
    load_dotenv()
    # Clean old tmp_code files
    for tmp_file in glob.glob("runs/coding/tmp_code_*.py"):
        try:
            os.remove(tmp_file)
        except OSError:
            pass

    parser = argparse.ArgumentParser(description="Run CaptainAgent on AssistantBench benchmark data.")
    parser.add_argument("--dataset", default="dataset/AssistantBench/assistant_bench_v1.0_dev.jsonl", help="Path to AssistantBench jsonl.")
    parser.add_argument("--mode", choices=["build_and_run", "run_only"], default="build_and_run")
    parser.add_argument("--build-name", default="assistantbench_captain", help="Logical name for the built team.")
    parser.add_argument(
        "--build-state",
        default=None,
        help="Path to store/load serialized build state. If omitted in build_and_run, a timestamped file will be generated.",
    )
    parser.add_argument(
        "--build-state-dir",
        default="runs/captain_builds",
        help="Directory to write timestamped build state files when --build-state is not provided.",
    )
    parser.add_argument("--work-dir", default="runs/coding", help="Working directory for code execution.")
    parser.add_argument("--output", default="runs/assistantbench/results.jsonl", help="Where to save model replies.")
    parser.add_argument(
        "--run-log-dir",
        default="runs/assistantbench_runs",
        help="Directory to store per-run artifacts (transcripts, summaries, runtime logs).",
    )
    parser.add_argument(
        "--runtime-log-dir",
        default="runs/runtime_logs",
        help="Directory to store autogen runtime logging sqlite DB.",
    )
    parser.add_argument(
        "--task-id",
        default=None,
        help="AssistantBench task_id to run.",
    )
    parser.add_argument(
        "--task-index",
        type=int,
        default=None,
        help="0-based index of task in the dataset (used when task_id is not provided).",
    )
    parser.add_argument(
        "--building-task",
        default=DEFAULT_BUILDING_TASK,
        help="Skill description used during the build phase.",
    )
    parser.add_argument(
        "--config-list",
        default="OAI_CONFIG_LIST",
        help="Path or env name for autogen config list. Defaults to OAI_CONFIG_LIST.",
    )
    parser.add_argument(
        "--model",
        default="qwen3_235b",
        help="Model name to use for filter/builder/agent.",
    )
    parser.add_argument("--max-round", type=int, default=30, help="Max nested group-chat rounds.")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    dataset_path = (repo_root / args.dataset).resolve()
    work_dir = (repo_root / args.work_dir).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d%H%M%S")
    task_label = args.task_id or f"assistantbench_task_{args.task_index}_{args.model}"
    run_dir = (repo_root / "runs" / f"{task_label}_{run_id}").resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    os.environ["LLM_STEP_DIR"] = str(run_dir / "llm_steps")

    if args.build_state:
        build_state_path = (repo_root / args.build_state).resolve()
    else:
        build_state_path = (run_dir / "build_state.json").resolve()

    task = _load_task(dataset_path, args.task_id, args.task_index)
    if not task:
        raise SystemExit(f"No AssistantBench task matched task_id={args.task_id} or task_index={args.task_index}.")

    general_llm_config, nested_mode_config = _build_llm_configs(args, repo_root)
    today = datetime.now().date().isoformat()

    meta_agent = MetaAgent(name="captain_agent", llm_config=general_llm_config, nested_mode="autobuild")

    def _is_terminate(msg: Dict) -> bool:
        content = msg.get("content", "") if isinstance(msg, dict) else ""
        return isinstance(content, str) and "terminate" in content.lower()

    meta_user_proxy = MetaUserProxyAgent(
        name="captain_user_proxy",
        nested_mode_config=nested_mode_config,
        code_execution_config={
            "use_docker": False,
            "work_dir": str(work_dir),
            "last_n_messages": 2,
        },
        default_group_name=args.build_name,
        is_termination_msg=_is_terminate,
        max_consecutive_auto_reply=1,
    )

    if args.mode == "run_only":
        if build_state_path is None or not build_state_path.exists():
            raise SystemExit("Build state not found. Provide --build-state pointing to an existing file.")
        meta_user_proxy.load_build_state(str(build_state_path))
    else:
        meta_user_proxy.build_team(args.build_name, args.building_task)
        meta_user_proxy.save_build_state(str(build_state_path), args.building_task)
        print(f"Build artifacts saved to: {build_state_path}", flush=True)

    answer, conversation_history, system_prompts = _run_single_task(
        meta_agent,
        meta_user_proxy,
        task.get("task", ""),
        today,
        max_round=args.max_round,
    )

    def _extract_answer(history: List[Dict], llm_step_dir: Path) -> str:
        def _extract_from_text(text: str) -> str:
            if not text or "## answer" not in text.lower():
                return ""
            _, remainder = text.split("## ANSWER", 1)
            lines = remainder.splitlines()
            for line in lines:
                stripped = line.strip()
                if not stripped:
                    continue
                if stripped.lower().startswith("## reason"):
                    break
                return stripped
            return ""

        # First try the cleaned conversation history so the final assistant
        # reply (which already includes ## ANSWER) is captured even if a later
        # judge step was appended afterwards.
        for msg in reversed(history):
            content = msg.get("content") if isinstance(msg, dict) else None
            if isinstance(content, str):
                extracted = _extract_from_text(content)
                if extracted:
                    return extracted

        # Fall back to scanning saved llm_step files (in reverse order).
        if llm_step_dir.exists():
            for path in sorted(llm_step_dir.glob("step_*.json"), reverse=True):
                try:
                    with path.open() as fh:
                        data = json.load(fh)
                except Exception:
                    continue
                # Responses are stored as raw strings; inspect both response and request.
                response_blob = data.get("response", "")
                if isinstance(response_blob, str):
                    extracted = _extract_from_text(response_blob)
                    if extracted:
                        return extracted
                req_messages = data.get("request", {}).get("messages", [])
                for msg in reversed(req_messages):
                    if isinstance(msg, dict):
                        content = msg.get("content")
                        if isinstance(content, str):
                            extracted = _extract_from_text(content)
                            if extracted:
                                return extracted
        return ""

    extracted_answer = _extract_answer(conversation_history, run_dir / "llm_steps")

    def _llm_judge_answer(ans: str, gt: str, llm_conf: Dict) -> Dict:
        prompt = f"""You are a strict judge. Given a ground truth answer and a model answer, decide if they match semantically.
Respond ONLY with JSON:
{{"is_correct": true/false, "reason": "brief justification"}}
Do not add extra text.
Ground truth: {gt}
Model answer: {ans}"""
        raw_reply = ""
        try:
            judge_agent = autogen.AssistantAgent(
                name="judge", llm_config=llm_conf, human_input_mode="NEVER", max_consecutive_auto_reply=1
            )
            raw_reply = judge_agent.generate_reply(messages=[{"role": "user", "content": prompt}])
            reply_text = raw_reply.get("content", "") if isinstance(raw_reply, dict) else str(raw_reply)
            reply_text = reply_text.replace("```json", "").replace("```", "").strip()
            match = re.search(r"(\{.*\})", reply_text, re.DOTALL)
            if match:
                reply_text = match.group(1).strip()
            if not reply_text:
                return {"is_correct": None, "reason": "No JSON object found in the reply", "raw_reply": reply_text}
            return json.loads(reply_text)
        except Exception as e:
            return {"is_correct": None, "reason": f"LLM judge failed: {e}", "raw_reply": str(raw_reply)}

    judge = {
        "ground_truth": task.get("answer"),
        "extracted_answer": extracted_answer,
    }
    judge.update(_llm_judge_answer(extracted_answer, task.get("answer") or "", general_llm_config))
    with (run_dir / "judge.json").open("w") as fh:
        json.dump(judge, fh, indent=2)

    # Extract agent information from agent_library.json
    agent_library_path = Path(__file__).parent.parent / "agent_library.json"
    agent_library = {}

    if agent_library_path.exists():
        with agent_library_path.open("r") as f:
            library_data = json.load(f)
            # Build a mapping of agent_name -> full agent info
            # Handle both dict with "agents" key and direct list format
            if isinstance(library_data, dict):
                agents_list = library_data.get("agents", [])
            elif isinstance(library_data, list):
                agents_list = library_data
            else:
                agents_list = []

            for agent_entry in agents_list:
                agent_name = agent_entry.get("name", "")
                if agent_name:
                    agent_library[agent_name] = {
                        "name": agent_name,
                        "description": agent_entry.get("description", ""),
                        "system_message": agent_entry.get("system_message", ""),
                        "path": agent_entry.get("path", "")
                    }

    # Extract agent info for all agents used in this task
    agents_info = {}
    for agent_name in system_prompts.keys():
        if agent_name in agent_library:
            # Use the original agent info from the library
            agents_info[agent_name] = agent_library[agent_name]
        else:
            # Agent not found in library (shouldn't happen in normal usage)
            agents_info[agent_name] = {
                "name": agent_name,
                "description": "",
                "system_message": "",
                "path": ""
            }

    # Generate comprehensive summary file
    summary = {
        "is_correct": judge.get("is_correct", False),
        "question": task.get("task", ""),
        "question_ID": task.get("id", ""),
        "difficulty": task.get("difficulty", ""),
        "ground_truth": task.get("answer", ""),
        "gold_url": task.get("gold_url", ""),
        "explanation": task.get("explanation", ""),
        "history": conversation_history,
        "agents": agents_info,
    }

    with (run_dir / "summary.json").open("w") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)

    print(f"Finished 1 task. Results saved to {run_dir}")
    print(f"Summary file generated: {run_dir / 'summary.json'}")
    print(f"Ground Truth:\n\n {task.get('answer')}")


if __name__ == "__main__":
    main()
