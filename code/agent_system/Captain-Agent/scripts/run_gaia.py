#!/usr/bin/env python
import argparse
import json
import os
import glob
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from dotenv import load_dotenv

import autogen
from autogen.agentchat.contrib.meta_agent import MetaAgent
from autogen.agentchat.contrib.meta_user_proxy_agent import MetaUserProxyAgent


GAIA_SYSTEM_TEMPLATE = """Today's date is {today}.

# Task
You need to solve the below question given by a user.

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

DEFAULT_BUILDING_TASK = """- Retrieval specialist skilled at web search, browsing and summarizing evidence
- Python coder who can read local files (csv, xlsx, pdf, image, audio) and verify answers with code
- Checker that re-runs verification and enforces grounding in cited evidence"""


def _load_task(dataset_path: Path, task_id: Optional[str], task_index: Optional[int]) -> Optional[Dict]:
    with dataset_path.open("r") as fh:
        records = [json.loads(line) for line in fh]
    if task_id:
        for record in records:
            if record.get("task_id") == task_id:
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
                # "env": {"PYTHONPATH": str(repo_root)},
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
    file_path: Optional[Path],
    max_round: int = 20,
) -> tuple[str, List[Dict], Dict]:
    prompt = GAIA_SYSTEM_TEMPLATE.format(today=today, question=question.strip(), max_round=max_round)
    if file_path:
        prompt = (
            f"Consider the local file '{file_path}'. If helpful, read it with python code blocks."
            f" Avoid asking the user to copy-paste content. {prompt}"
        )
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

    parser = argparse.ArgumentParser(description="Run CaptainAgent on GAIA benchmark data without autogenbench.")
    parser.add_argument("--dataset", default="dataset/gaia-val/standardized_data.jsonl", help="Path to GAIA jsonl.")
    parser.add_argument("--mode", choices=["build_and_run", "run_only"], default="build_and_run")
    parser.add_argument("--build-name", default="gaia_captain", help="Logical name for the built team.")
    parser.add_argument("--work-dir", default="runs/coding", help="Working directory for code execution.")
    parser.add_argument(
        "--task-id",
        default=None,
        help="GAIA task_id to run.",
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
    parser.add_argument("--max-round", type=int, default=20, help="Max nested group-chat rounds.")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    dataset_path = (repo_root / args.dataset).resolve()
    work_dir = (repo_root / args.work_dir).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)

    task = _load_task(dataset_path, args.task_id, args.task_index)
    if not task:
        raise SystemExit(f"No GAIA task matched task_id={args.task_id} or task_index={args.task_index}.")

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
            # "env": {"PYTHONPATH": str(repo_root)},
        },
        default_group_name=args.build_name,
        is_termination_msg=_is_terminate,
        max_consecutive_auto_reply=1,
    )

    if args.mode == "run_only":
        raise SystemExit("run_only mode is not supported without build state.")
    else:
        meta_user_proxy.build_team(args.build_name, args.building_task)

    file_path = task.get("file_path")
    resolved_file = (dataset_path.parent / file_path).resolve() if file_path else None
    answer, conversation_history, system_prompts = _run_single_task(
        meta_agent,
        meta_user_proxy,
        task.get("task_question", ""),
        today,
        resolved_file,
        max_round=args.max_round,
    )

    print(f"Finished 1 task.")
    print(f"Ground Truth: {task.get('ground_truth')}")


if __name__ == "__main__":
    main()
