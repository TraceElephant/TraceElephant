import os
import json
import random
import re
from pathlib import Path
from openai import AzureOpenAI
from tqdm import tqdm


class ContextLengthExceeded(Exception):
    pass

# --- Helper Functions ---

def _is_raw_test_data_format(directory_path):
    """
    Check if directory is in raw_test_data_ca format (contains subdirectories instead of direct JSON files)
    """
    dir_path = Path(directory_path)
    if not dir_path.exists():
        return False

    # Check for captain-runs-*, magentic-runs-*, swe-agent-runs-* subdirectories
    has_captain_runs = any(d.name.startswith('captain-runs-') or d.name.startswith('magentic-runs-')
                           or d.name.startswith('swe-agent-runs-') for d in dir_path.iterdir() if d.is_dir())
    return has_captain_runs


def _load_task_from_directory(task_dir):
    """
    Load data from task directory in raw_test_data_ca format
    Supports two formats:
    1. Old format: summary.json + history.json
    2. New format: trace_metadata.json + step_records.json

    Args:
        task_dir: Path object pointing to task directory

    Returns:
        Dict containing: question, ground_truth, history, mistake_agent, mistake_step
    """
    # Check new format
    metadata_file = task_dir / "trace_metadata.json"
    step_records_file = task_dir / "step_records.json"

    # Check old format
    summary_file = task_dir / "summary.json"
    history_file = task_dir / "history.json"

    # Try loading new format
    if metadata_file.exists() and step_records_file.exists():
        return _load_task_from_new_format(task_dir, metadata_file, step_records_file)

    # Try loading old format
    elif summary_file.exists() and history_file.exists():
        return _load_task_from_old_format(task_dir, summary_file, history_file)

    else:
        return None


def _load_task_from_new_format(task_dir, metadata_file, step_records_file):
    """
    Load task data from new format (trace_metadata.json + step_records.json)

    Args:
        task_dir: Path object pointing to task directory
        metadata_file: trace_metadata.json file path
        step_records_file: step_records.json file path

    Returns:
        Dict containing: question, ground_truth, history, mistake_agent, mistake_step
    """
    try:
        with open(metadata_file, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        with open(step_records_file, 'r', encoding='utf-8') as f:
            step_records = json.load(f)

        # Convert step_records to history format
        converted_history = []
        for step in step_records:
            agent_name = step.get('agent_name', 'Unknown')
            step_input = step.get('input', {})
            step_output = step.get('output', '')

            # Build request and response
            request_payload = step_input

            # Try parsing output string as ChatCompletion object
            # output format looks like: "ChatCompletion(id='...', choices=[...])"
            response_payload = {
                'raw_output': step_output
            }

            # Try extracting actual message content from output
            # Look for message=ChatCompletionMessage(content='...', ...) pattern
            try:
                if isinstance(step_output, str) and 'content=' in step_output:
                    # Use regex to extract content
                    import re
                    content_match = re.search(r"content='(.*?)'(?:,|\))", step_output.replace("\\'", "'"))
                    if content_match:
                        actual_content = content_match.group(1)
                        # Handle escape characters
                        actual_content = actual_content.replace('\\n', '\n').replace('\\t', '\t')
                        response_payload['content'] = actual_content
            except:
                pass

            content = json.dumps(
                {"request": request_payload, "response": response_payload},
                ensure_ascii=True
            )

            converted_history.append({
                "name": agent_name,
                "content": content,
                "request": request_payload,
                "response": response_payload
            })

        return {
            "question": metadata.get("task_instruction", ""),
            "ground_truth": metadata.get("ground_truth", ""),
            "tests_status": metadata.get("tests_status", None),
            "history": converted_history,
            "mistake_agent": metadata.get("mistake_agent", ""),
            "mistake_step": str(metadata.get("mistake_step", "")),
            "agent_system_intro": metadata.get("agent_system_intro", "")
        }
    except Exception as e:
        print(f"Error loading task from new format {task_dir}: {e}")
        import traceback
        traceback.print_exc()
        return None


def _load_task_from_old_format(task_dir, summary_file, history_file):
    """
    Load task data from old format (summary.json + history.json)

    Args:
        task_dir: Path object pointing to task directory
        summary_file: summary.json file path
        history_file: history.json file path

    Returns:
        Dict containing: question, ground_truth, history, mistake_agent, mistake_step
    """
    try:
        with open(summary_file, 'r', encoding='utf-8') as f:
            summary = json.load(f)
        with open(history_file, 'r', encoding='utf-8') as f:
            history_steps = json.load(f)

        # Convert history format
        converted_history = []
        for step in history_steps:
            agent_name = step.get('agent_name', 'Unknown')

            # Use request + response as step content
            request_payload = step.get('request', {})
            response_payload = step.get('response', {})
            content = json.dumps(
                {"request": request_payload, "response": response_payload},
                ensure_ascii=True
            )

            converted_history.append({
                "name": agent_name,
                "content": content,
                "request": request_payload,
                "response": response_payload
            })

        return {
            "question": summary.get("question", ""),
            "ground_truth": summary.get("ground_truth", ""),
            "tests_status": summary.get("tests_status", None),
            "history": converted_history,
            "mistake_agent": summary.get("mistake_agent", ""),
            "mistake_step": summary.get("mistake_step", ""),
            "agent_system_intro": summary.get("agent_system_intro", "")
        }
    except Exception as e:
        print(f"Error loading task from old format {task_dir}: {e}")
        return None


def _get_task_data_list(directory_path):
    """
    Get task data list, compatible with two formats:
    1. Single directory containing multiple JSON files (original format)
    2. raw_test_data_ca format (subdirectory structure)

    Returns:
        List of (task_name, task_data) tuples
    """
    dir_path = Path(directory_path)

    if _is_raw_test_data_format(directory_path):
        # raw_test_data_ca format or newest_data format
        print(f"Detected structured directory format")
        task_data_list = []

        # Support multiple directory prefixes
        patterns = ["captain-runs-*", "magentic-runs-*", "swe-agent-runs-*"]
        # patterns = ["captain-runs-*", "magentic-runs-*"]

        for pattern in patterns:
            for runs_dir in sorted(dir_path.glob(pattern)):
                if not runs_dir.is_dir():
                    continue

                for task_dir in sorted(runs_dir.iterdir()):
                    if not task_dir.is_dir():
                        continue

                    task_data = _load_task_from_directory(task_dir)
                    if task_data:
                        task_data_list.append((task_dir.name, task_data))

        return task_data_list
    else:
        # Original format: JSON files in directory
        print(f"Detected original JSON file format")
        json_files = _get_sorted_json_files(directory_path)
        task_data_list = []

        for json_file in json_files:
            file_path = os.path.join(directory_path, json_file)
            data = _load_json_data(file_path)
            if data:
                task_data_list.append((json_file, data))

        return task_data_list


# --- Original Helper Functions ---

def _get_sorted_json_files(directory_path):
    """Gets and sorts JSON files numerically from a directory."""
    try:
        files = [f for f in os.listdir(directory_path) if f.endswith('.json')]
        return sorted(files, key=lambda x: int(''.join(filter(str.isdigit, x)) or 0))
    except FileNotFoundError:
        print(f"Error: Directory not found at {directory_path}")
        return []
    except Exception as e:
        print(f"Error reading or sorting files in {directory_path}: {e}")
        return []

def _load_json_data(file_path):
    """Loads data from a JSON file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {file_path}")
        return None
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return None

def _make_api_call(client, model, messages, max_tokens):
    """Makes an API call to Azure OpenAI."""
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            timeout=60
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        error_text = str(e)
        if "context_length_exceeded" in error_text or "maximum context length" in error_text:
            raise ContextLengthExceeded(error_text)
        print(f"Error during OpenAI API call: {e}")
        return None


def _get_entry_content(entry, content_mode: str) -> str:
    if content_mode == "full":
        if "request" in entry or "response" in entry:
            payload = {
                "request": entry.get("request", {}),
                "response": entry.get("response", {})
            }
            return json.dumps(payload, ensure_ascii=True)
        return entry.get("content", "")

    if content_mode == "truncate_middle":
        # New strategy: when request is too long, keep beginning and end, replace middle with ...
        if "request" in entry or "response" in entry:
            request_payload = entry.get("request", {})
            response_payload = entry.get("response", {})

            # Serialize request to check length
            request_str = json.dumps(request_payload, ensure_ascii=True)

            MAX_REQUEST_LENGTH = 3000
            MAX_REQUEST_LENGTH_HALF = MAX_REQUEST_LENGTH // 2

            # If request exceeds 30k characters, truncate
            if len(request_str) > MAX_REQUEST_LENGTH:
                # Keep first 15k and last 15k characters
                truncated_request_str = request_str[:MAX_REQUEST_LENGTH_HALF] + "\n...[truncated middle content]...\n" + request_str[-MAX_REQUEST_LENGTH_HALF:]
                # Try parsing back to JSON (may fail, use string form if so)
                try:
                    # Use truncated string representation directly
                    payload = {
                        "request": f"[TRUNCATED] {truncated_request_str}",
                        "response": response_payload
                    }
                except:
                    payload = {
                        "request": request_payload,
                        "response": response_payload
                    }
            else:
                payload = {
                    "request": request_payload,
                    "response": response_payload
                }

            return json.dumps(payload, ensure_ascii=True)
        return entry.get("content", "")

    if content_mode == "request_last":
        request_payload = entry.get("request")
        response_payload = entry.get("response", {})
        if isinstance(request_payload, dict):
            req_copy = dict(request_payload)
            messages = req_copy.get("messages")
            if isinstance(messages, list) and messages:
                req_copy["messages"] = [messages[-1]]
            payload = {"request": req_copy, "response": response_payload}
            return json.dumps(payload, ensure_ascii=True)
        if "response" in entry:
            return json.dumps(response_payload, ensure_ascii=True)
        return entry.get("content", "")

    if content_mode == "response_only":
        if "response" in entry:
            return json.dumps(entry.get("response", {}), ensure_ascii=True)
        return entry.get("content", "")

    return entry.get("content", "")


def _summarize_messages(client, model, messages, max_tokens):
    if not messages:
        return ""
    summary_input = []
    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        summary_input.append(f"{role}: {content}")
    summary_text = "\n".join(summary_input)
    summary_messages = [
        {"role": "system", "content": "Summarize the following messages concisely, preserving key facts."},
        {"role": "user", "content": summary_text},
    ]
    try:
        return _make_api_call(client, model, summary_messages, max_tokens) or ""
    except ContextLengthExceeded:
        return ""


def _format_chat_history_with_summary(chat_history, index_agent, client, model, start_index=0, summary_max_tokens=256):
    lines = []
    for offset, entry in enumerate(chat_history):
        step_number = start_index + offset + 1
        agent_name = entry.get(index_agent, 'Unknown Agent')
        request_payload = entry.get("request", {})
        response_payload = entry.get("response", {})

        if isinstance(request_payload, dict):
            req_copy = dict(request_payload)
            messages = req_copy.get("messages", [])
            last_msg = messages[-1] if isinstance(messages, list) and messages else None

            summary = entry.get("_request_summary")
            if summary is None:
                summary = _summarize_messages(client, model, messages[:-1] if last_msg else messages, summary_max_tokens)
                entry["_request_summary"] = summary

            summarized_messages = []
            if summary:
                summarized_messages.append({
                    "role": "system",
                    "content": f"Summary of previous messages: {summary}"
                })
            if last_msg:
                summarized_messages.append(last_msg)

            if summarized_messages:
                req_copy["messages"] = summarized_messages

            payload = {"request": req_copy, "response": response_payload}
            content = json.dumps(payload, ensure_ascii=True)
        else:
            content = _get_entry_content(entry, "full")

        lines.append(f"Step {step_number} - {agent_name}: {content}")
    return "\n".join(lines)
    
def _format_chat_history(chat_history, index_agent, start_index=0, content_mode="full"):
    lines = []
    for offset, entry in enumerate(chat_history):
        step_number = start_index + offset + 1
        agent_name = entry.get(index_agent, 'Unknown Agent')
        content = _get_entry_content(entry, content_mode)
        lines.append(f"Step {step_number} - {agent_name}: {content}")
    return "\n".join(lines)



def _try_api_call_with_fallback(client, model, max_tokens, message_builders, task_name=None):
    """
    message_builders: list of (label, callable -> messages)
    """
    for label, builder in message_builders:
        try:
            return _make_api_call(client, model, builder(), max_tokens)
        except ContextLengthExceeded as e:
            if task_name:
                print(f"Context length exceeded for {task_name} using {label}. Retrying with shorter input.")
            else:
                print(f"Context length exceeded using {label}. Retrying with shorter input.")
            continue
    return None

def _format_tests_status(tests_status):
    """
    Format test status information into readable string

    Args:
        tests_status: Test status dictionary containing FAIL_TO_PASS, PASS_TO_PASS, etc.

    Returns:
        Formatted string
    """
    if not tests_status:
        return ""

    lines = ["Test Status Information:"]

    for category, results in tests_status.items():
        if category in ["FAIL_TO_PASS", "PASS_TO_PASS", "FAIL_TO_FAIL", "PASS_TO_FAIL"]:
            success_tests = results.get("success", [])
            failure_tests = results.get("failure", [])

            if success_tests or failure_tests:
                lines.append(f"\n{category}:")
                if success_tests:
                    lines.append(f"  Success ({len(success_tests)} tests): {', '.join(success_tests[:3])}" +
                               (" ..." if len(success_tests) > 3 else ""))
                if failure_tests:
                    lines.append(f"  Failure ({len(failure_tests)} tests): {', '.join(failure_tests[:3])}" +
                               (" ..." if len(failure_tests) > 3 else ""))

    return "\n".join(lines)


def _build_all_at_once_prompt(problem, ground_truth, chat_content, tests_status=None, agent_system_intro=""):
    # Prefer tests_status, fallback to ground_truth if unavailable
    info_line = ""
    if tests_status:
        info_line = _format_tests_status(tests_status) + "\n"
    elif ground_truth:
        info_line = f"The Answer for the problem is: {ground_truth}\n"

    # Add agent system intro
    agent_intro_line = ""
    if agent_system_intro:
        agent_intro_line = f"\nAgent System Introduction: {agent_system_intro}\nThis is an introduction to the agent system that generated this trace.\n\n"

    return (
        "You are an AI assistant tasked with analyzing a multi-agent conversation history when solving a real world problem. "
        f"The problem is:  {problem}\n"
        f"{info_line}"
        f"{agent_intro_line}"
        "Identify which agent made an error, at which step, and explain the reason for the error. "
        "Here's the conversation (each line is a step, starting from 1):\n\n" + chat_content +
        "\n\nBased on this conversation, please predict the following:\n"
        "1. The name of the agent who made a mistake that should be directly responsible for the wrong solution to the real world problem. If there are no agents that make obvious mistakes, decide one single agent in your mind. Directly output the name of the Expert.\n"
        "2. The step number (starting from 1) where the mistake agent first made a mistake. The conversation is formatted as:\n"
        "Step 1 - agent a: xx\n"
        "Step 2 - agent b: xxxx\n"
        "Step 3 - agent c: xxxxx\n"
        "Step 4 - agent a: xxxxxxx\n"
        "Please determine the step number where the first mistake occurred.\n"
        "3. The reason for your prediction."
        "Please answer with a JSON block wrapped in triple backticks, in this exact schema:\n"
        "```json\n"
        "{\"agent_name\": \"...\", \"step_number\": 0, \"reason\": \"...\"}\n"
        "```\n"
    )

def _build_step_by_step_prompt(problem, ground_truth, current_conversation_history, idx, agent_name, tests_status=None, agent_system_intro=""):
    # Prefer tests_status, fallback to ground_truth if unavailable
    info_line = ""
    if tests_status:
        info_line = _format_tests_status(tests_status) + "\n"
    elif ground_truth:
        info_line = f"The Answer for the problem is: {ground_truth}\n"

    # Add agent system intro
    agent_intro_line = ""
    if agent_system_intro:
        agent_intro_line = f"\nAgent System Introduction: {agent_system_intro}\nThis is an introduction to the agent system that generated this trace.\n\n"

    display_idx = idx + 1
    return (
        f"You are an AI assistant tasked with evaluating the correctness of each step in an ongoing multi-agent conversation aimed at solving a real-world problem. The problem being addressed is: {problem}. "
        f"{info_line}"
        f"{agent_intro_line}"
        f"Here is the conversation history up to the current step (steps start from 1):\n{current_conversation_history}\n"
        f"The most recent step ({display_idx}) was by '{agent_name}'.\n"
        f"Your task is to determine whether this most recent agent's action (Step {display_idx}) contains an error that could hinder the problem-solving process or lead to an incorrect solution. "
        "Please respond with 'Yes' or 'No' and provide a clear explanation for your judgment. "
        "Note: Please avoid being overly critical in your evaluation. Focus on errors that clearly derail the process."
        "Respond ONLY in the format: 1. Yes/No.\n2. Reason: [Your explanation here]"
    )

def _normalize_task_id(task_name):
    return task_name.replace(".json", "")

def _parse_prediction_from_text(text):
    if not text:
        return None, None, None

    json_block_match = re.search(r"```json\s*(\{.*?\})\s*```", text, re.IGNORECASE | re.DOTALL)
    if json_block_match:
        try:
            data = json.loads(json_block_match.group(1))
            agent = str(data.get("agent_name", "")).strip() or None
            step_val = data.get("step_number", "")
            step = str(step_val).strip() if step_val != "" else None
            reason = str(data.get("reason", "")).strip() or None
            return agent, step, reason
        except Exception:
            pass

    agent_match = re.search(r"Agent Name:\s*([^\n\r]+)", text, re.IGNORECASE)
    step_match = re.search(r"Step Number:\s*(\d+)", text, re.IGNORECASE)
    reason_match = re.search(r"Reason(?: for Mistake)?:\s*(.*)", text, re.IGNORECASE | re.DOTALL)

    agent = agent_match.group(1).strip() if agent_match else None
    step = step_match.group(1).strip() if step_match else None
    reason = reason_match.group(1).strip() if reason_match else None
    return agent, step, reason

# --- All-at-Once Method ---

def all_at_once(client: AzureOpenAI, directory_path: str, is_handcrafted: bool, model: str, max_tokens: int):
    """
    Analyzes chat history by feeding the entire conversation at once to the model.
    Supports both original JSON format and raw_test_data_ca format.
    """
    print("\n--- Starting All-at-Once Analysis ---\n")

    # Use new unified data loading function
    task_data_list = _get_task_data_list(directory_path)
    index_agent = "role" if is_handcrafted else "name"

    results = []

    for task_name, data in tqdm(task_data_list):
        if not data:
            results.append({
                "task_id": _normalize_task_id(task_name),
                "predicted_agent": "",
                "predicted_step": "",
                "reason": "",
                "raw_response": "",
                "success": False
            })
            continue

        chat_history = data.get("history", [])
        problem = data.get("question", "")
        ground_truth = data.get("ground_truth", "")
        tests_status = data.get("tests_status", None)
        agent_system_intro = data.get("agent_system_intro", "")

        if not chat_history:
            print(f"Skipping {task_name}: No chat history found.")
            results.append({
                "task_id": _normalize_task_id(task_name),
                "predicted_agent": "",
                "predicted_step": "",
                "reason": "",
                "raw_response": "",
                "success": False
            })
            continue

        def build_messages(content_mode):
            chat_content = _format_chat_history(
                chat_history,
                index_agent,
                start_index=0,
                content_mode=content_mode
            )
            prompt = _build_all_at_once_prompt(problem, ground_truth, chat_content, tests_status, agent_system_intro)
            return [
                {"role": "system", "content": "You are a helpful assistant skilled in analyzing conversations."},
                {"role": "user", "content": prompt},
            ]

        def build_messages_summary():
            chat_content = _format_chat_history_with_summary(
                chat_history,
                index_agent,
                client,
                model,
                start_index=0,
                summary_max_tokens=256
            )
            prompt = _build_all_at_once_prompt(problem, ground_truth, chat_content, tests_status, agent_system_intro)
            return [
                {"role": "system", "content": "You are a helpful assistant skilled in analyzing conversations."},
                {"role": "user", "content": prompt},
            ]

        result = _try_api_call_with_fallback(
            client,
            model,
            max_tokens,
            [
                ("full", lambda: build_messages("full")),
                ("truncate_middle", lambda: build_messages("truncate_middle")),
                ("response_only", lambda: build_messages("response_only")),
            ],
            task_name=task_name
        )
        agent, step, reason = _parse_prediction_from_text(result)
        success = bool(result and agent and step)

        print(f"Prediction for {task_name}:")
        if result:
            print(result)
        else:
            print("Failed to get prediction.")
        print("\n" + "="*50 + "\n")

        results.append({
            "task_id": _normalize_task_id(task_name),
            "predicted_agent": agent or "",
            "predicted_step": step or "",
            "reason": reason or "",
            "raw_response": result or "",
            "success": success
        })

    return results

# --- Step-by-Step Method ---

def step_by_step(client: AzureOpenAI, directory_path: str, is_handcrafted: bool, model: str, max_tokens: int):
    """
    Analyzes chat history step by step, asking the model at each step if an error occurred.
    Supports both original JSON format and raw_test_data_ca format.
    """
    print("\n--- Starting Step-by-Step Analysis ---\n")

    # Use new unified data loading function
    task_data_list = _get_task_data_list(directory_path)
    index_agent = "role" if is_handcrafted else "name"

    results = []

    for task_name, data in tqdm(task_data_list):
        if not data:
            results.append({
                "task_id": _normalize_task_id(task_name),
                "predicted_agent": "",
                "predicted_step": "",
                "reason": "",
                "raw_response": "",
                "success": False
            })
            continue

        chat_history = data.get("history", [])
        problem = data.get("question", "")
        ground_truth = data.get("ground_truth", "")
        tests_status = data.get("tests_status", None)
        agent_system_intro = data.get("agent_system_intro", "")

        if not chat_history:
            print(f"Skipping {task_name}: No chat history found.")
            results.append({
                "task_id": _normalize_task_id(task_name),
                "predicted_agent": "",
                "predicted_step": "",
                "reason": "",
                "raw_response": "",
                "success": False
            })
            continue

        print(f"--- Analyzing Task: {task_name} ---")
        current_conversation_history = ""
        error_found = False
        predicted_agent = ""
        predicted_step = ""
        reason = ""
        last_answer = ""
        for idx, entry in enumerate(chat_history):
            agent_name = entry.get(index_agent, 'Unknown Agent')

            def build_messages(content_mode):
                current_conversation_history = _format_chat_history(
                    chat_history[: idx + 1],
                    index_agent,
                    start_index=0,
                    content_mode=content_mode
                )
                prompt = _build_step_by_step_prompt(
                    problem,
                    ground_truth,
                    current_conversation_history,
                    idx,
                    agent_name,
                    tests_status,
                    agent_system_intro
                )
                return [
                    {"role": "system", "content": "You are a precise step-by-step conversation evaluator."},
                    {"role": "user", "content": prompt},
                ]

            print(f"Evaluating Step {idx + 1} by {agent_name}...")
            def build_messages_summary():
                current_conversation_history = _format_chat_history_with_summary(
                    chat_history[: idx + 1],
                    index_agent,
                    client,
                    model,
                    start_index=0,
                    summary_max_tokens=256
                )
                prompt = _build_step_by_step_prompt(
                    problem,
                    ground_truth,
                    current_conversation_history,
                    idx,
                    agent_name,
                    tests_status,
                    agent_system_intro
                )
                return [
                    {"role": "system", "content": "You are a precise step-by-step conversation evaluator."},
                    {"role": "user", "content": prompt},
                ]

            answer = _try_api_call_with_fallback(
                client,
                model,
                max_tokens,
                [
                    ("full", lambda: build_messages("full")),
                    ("truncate_middle", lambda: build_messages("truncate_middle")),
                    ("response_only", lambda: build_messages("response_only")),
                ],
                task_name=task_name
            )
            last_answer = answer or ""

            if not answer:
                print("Failed to get evaluation for this step. Stopping analysis for this file.")
                error_found = True # Treat API error as unable to proceed
                break

            print(f"LLM Evaluation: {answer}")

            # Basic check for "Yes" at the beginning of the response
            if answer.lower().strip().startswith("1. yes"):
                print(f"\nPrediction for {task_name}: Error found.")
                print(f"Agent Name: {agent_name}")
                print(f"Step Number: {idx + 1}")
                print(f"Reason provided by LLM: {answer.split('Reason:', 1)[-1].strip()}")
                predicted_agent = agent_name
                predicted_step = str(idx + 1)
                reason = answer.split('Reason:', 1)[-1].strip()
                error_found = True
                break  # Stop processing this file once an error is found
            elif answer.lower().strip().startswith("1. no"):
                 print("No significant error detected in this step.")
            else:
                print("Warning: Unexpected response format from LLM. Continuing evaluation.")
                # Optionally handle unexpected format more robustly

        if not error_found:
            print(f"\nNo decisive errors found by step-by-step analysis in task {task_name}")

        print("\n" + "="*50 + "\n")

        results.append({
            "task_id": _normalize_task_id(task_name),
            "predicted_agent": predicted_agent,
            "predicted_step": predicted_step,
            "reason": reason,
            "raw_response": last_answer,
            "success": bool(predicted_agent and predicted_step)
        })

    return results


# --- Binary Search Method ---

def _construct_binary_search_prompt(problem, answer, chat_segment_content, range_description, upper_half_desc, lower_half_desc, tests_status=None, agent_system_intro=""):
    """Constructs the prompt for the binary search step."""
    # Prefer tests_status, fallback to ground_truth (answer parameter) if unavailable
    info_line = ""
    if tests_status:
        info_line = _format_tests_status(tests_status) + "\n"
    elif answer:
        info_line = f"The Answer for the problem is: {answer}\n"

    # Add agent system intro
    agent_intro_line = ""
    if agent_system_intro:
        agent_intro_line = f"\nAgent System Introduction: {agent_system_intro}\nThis is an introduction to the agent system that generated this trace.\n\n"

    return (
        "You are an AI assistant tasked with analyzing a segment of a multi-agent conversation. Multiple agents are collaborating to address a user query, with the goal of resolving the query through their collective dialogue.\n"
        "Your primary task is to identify the location of the most critical mistake within the provided segment. Determine which half of the segment contains the single step where this crucial error occurs, ultimately leading to the failure in resolving the user's query.\n"
        f"The problem to address is as follows: {problem}\n"
        f"{info_line}"
        f"{agent_intro_line}"
        f"Review the following conversation segment {range_description} (steps start from 1):\n\n{chat_segment_content}\n\n"
        f"Based on your analysis, predict whether the most critical error is more likely to be located in the upper half ({upper_half_desc}) or the lower half ({lower_half_desc}) of this segment.\n"
        "Please provide your prediction by responding with ONLY 'upper half' or 'lower half'. Remember, your answer should be based on identifying the mistake that directly contributes to the failure in resolving the user's query. If no single clear error is evident, consider the step you believe is most responsible for the failure, allowing for subjective judgment, and base your answer on that."
    )

def _report_binary_search_error(chat_history, step, task_name, is_handcrafted):
    """Reports the identified error step from binary search."""
    index_agent = "role" if is_handcrafted else "name"
    entry = chat_history[step]
    agent_name = entry.get(index_agent, 'Unknown Agent')

    print(f"\nPrediction for {task_name}:")
    print(f"Agent Name: {agent_name}")
    print(f"Step Number: {step + 1}")
    print("\n" + "="*50 + "\n")
    return step, agent_name

def _find_error_in_segment_recursive(client: AzureOpenAI, model: str, max_tokens: int, chat_history: list, problem: str, answer: str, start: int, end: int, task_name: str, is_handcrafted: bool, tests_status=None, agent_system_intro=""):
    """Recursive helper function for binary search analysis."""
    if start > end:
         print(f"Warning: Invalid range in binary search for {task_name} (start={start}, end={end}). Reporting last valid step.")
         step, agent_name = _report_binary_search_error(chat_history, end if end >= 0 else 0, task_name, is_handcrafted)
         return step, agent_name, ""
    if start == end:
        step, agent_name = _report_binary_search_error(chat_history, start, task_name, is_handcrafted)
        return step, agent_name, ""

    index_agent = "role" if is_handcrafted else "name"

    segment_history = chat_history[start : end + 1]
    if not segment_history:
        print(f"Warning: Empty segment in binary search for {task_name} (start={start}, end={end}). Cannot proceed.")
        step, agent_name = _report_binary_search_error(chat_history, start, task_name, is_handcrafted)
        return step, agent_name, ""

    mid = start + (end - start) // 2

    range_description = f"from step {start + 1} to step {end + 1}"
    upper_half_desc = f"from step {start + 1} to step {mid + 1}"
    lower_half_desc = f"from step {mid + 2} to step {end + 1}"

    def build_messages(content_mode):
        chat_content = _format_chat_history(
            segment_history,
            index_agent,
            start_index=start,
            content_mode=content_mode
        )
        prompt = _construct_binary_search_prompt(
            problem,
            answer,
            chat_content,
            range_description,
            upper_half_desc,
            lower_half_desc,
            tests_status,
            agent_system_intro
        )
        return [
            {"role": "system", "content": "You are an AI assistant specializing in localizing errors in conversation segments."},
            {"role": "user", "content": prompt}
        ]

    print(f"Analyzing step {start + 1}-{end + 1} for {task_name}...")
    def build_messages_summary():
        chat_content = _format_chat_history_with_summary(
            segment_history,
            index_agent,
            client,
            model,
            start_index=start,
            summary_max_tokens=256
        )
        prompt = _construct_binary_search_prompt(
            problem,
            answer,
            chat_content,
            range_description,
            upper_half_desc,
            lower_half_desc,
            tests_status,
            agent_system_intro
        )
        return [
            {"role": "system", "content": "You are an AI assistant specializing in localizing errors in conversation segments."},
            {"role": "user", "content": prompt}
        ]

    result = _try_api_call_with_fallback(
        client,
        model,
        max_tokens,
        [
            ("full", lambda: build_messages("full")),
            ("truncate_middle", lambda: build_messages("truncate_middle")),
            ("response_only", lambda: build_messages("response_only")),
        ],
        task_name=task_name
    )

    if not result:
        print(f"API call failed for segment {start}-{end}. Stopping binary search for {task_name}.")
        return None, "", ""

    print(f"LLM Prediction for segment {start}-{end}: {result}")
    result_lower = result.lower() 

    if "upper half" in result_lower:
         return _find_error_in_segment_recursive(client, model, max_tokens, chat_history, problem, answer, start, mid, task_name, is_handcrafted, tests_status, agent_system_intro)
    elif "lower half" in result_lower:
         new_start = min(mid + 1, end)
         return _find_error_in_segment_recursive(client, model, max_tokens, chat_history, problem, answer, new_start, end, task_name, is_handcrafted, tests_status, agent_system_intro)
    else:
        print(f"Warning: Ambiguous response '{result}' from LLM for segment {start}-{end}. Randomly choosing a half.")
        if random.randint(0, 1) == 0:
            print("Randomly chose upper half.")
            return _find_error_in_segment_recursive(client, model, max_tokens, chat_history, problem, answer, start, mid, task_name, is_handcrafted, tests_status, agent_system_intro)
        else:
            print("Randomly chose lower half.")
            new_start = min(mid + 1, end)
            return _find_error_in_segment_recursive(client, model, max_tokens, chat_history, problem, answer, new_start, end, task_name, is_handcrafted, tests_status, agent_system_intro)


def binary_search(client: AzureOpenAI, directory_path: str, is_handcrafted: bool, model: str, max_tokens: int):
    """
    Analyzes chat history using a binary search approach to find the error step.
    Supports both original JSON format and raw_test_data_ca format.
    """
    print("\n--- Starting Binary Search Analysis ---\n")

    # Use new unified data loading function
    task_data_list = _get_task_data_list(directory_path)

    results = []

    for task_name, data in tqdm(task_data_list):
        if not data:
            results.append({
                "task_id": _normalize_task_id(task_name),
                "predicted_agent": "",
                "predicted_step": "",
                "reason": "",
                "raw_response": "",
                "success": False
            })
            continue

        chat_history = data.get("history", [])
        problem = data.get("question", "")
        answer = data.get("ground_truth", "")
        tests_status = data.get("tests_status", None)
        agent_system_intro = data.get("agent_system_intro", "")

        if not chat_history:
            print(f"Skipping {task_name}: No chat history found.")
            results.append({
                "task_id": _normalize_task_id(task_name),
                "predicted_agent": "",
                "predicted_step": "",
                "reason": "",
                "raw_response": "",
                "success": False
            })
            continue

        print(f"--- Analyzing Task: {task_name} ---")
        predicted_step, predicted_agent, _ = _find_error_in_segment_recursive(
            client, model, max_tokens, chat_history, problem, answer, 0, len(chat_history) - 1, task_name, is_handcrafted, use_ground_truth, tests_status, agent_system_intro
        )
        results.append({
            "task_id": _normalize_task_id(task_name),
            "predicted_agent": predicted_agent or "",
            "predicted_step": str(predicted_step + 1) if predicted_step is not None else "",
            "reason": "",
            "raw_response": "",
            "success": bool(predicted_agent and predicted_step is not None)
        })

    return results
