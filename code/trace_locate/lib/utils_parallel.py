"""
Parallel version of inference tools
Supports multi-threaded parallel task processing to significantly improve inference speed
"""

import os
import json
import random
from pathlib import Path
from openai import AzureOpenAI, OpenAI
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# Import original helper functions
from .utils import (
    _get_task_data_list,
    _normalize_task_id,
    _parse_prediction_from_text,
    _build_all_at_once_prompt,
    _build_step_by_step_prompt,
    _construct_binary_search_prompt,
    _format_chat_history,
    _try_api_call_with_fallback,
    _format_chat_history_with_summary
)

# Thread lock for protecting output
output_lock = threading.Lock()


def _process_task_all_at_once(client, model, max_tokens, task_name, data):
    """
    Process single task - all_at_once method

    Args:
        client: OpenAI client
        model: Model name
        max_tokens: Maximum token count
        task_name: Task name
        data: Task data

    Returns:
        (task_name, result_text, success)
    """
    if not data:
        return {
            "task_id": _normalize_task_id(task_name),
            "predicted_agent": "",
            "predicted_step": "",
            "reason": "",
            "raw_response": "",
            "success": False
        }

    chat_history = data.get("history", [])
    problem = data.get("question", "")
    ground_truth = data.get("ground_truth", "")
    tests_status = data.get("tests_status", None)
    agent_system_intro = data.get("agent_system_intro", "")

    if not chat_history:
        return {
            "task_id": _normalize_task_id(task_name),
            "predicted_agent": "",
            "predicted_step": "",
            "reason": "",
            "raw_response": "",
            "success": False
        }

    index_agent = "name"

    try:
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

        result = _try_api_call_with_fallback(
            client,
            model,
            max_tokens,
            [
                ("full", lambda: build_messages("full")),
                # ("truncate_middle", lambda: build_messages("truncate_middle")),
                ("response_only", lambda: build_messages("response_only")),
            ],
            task_name=task_name
        )

        agent, step, reason = _parse_prediction_from_text(result) if result else ("", "", "")

        final_response = result or ""
        success = bool(result and agent and step)
        return {
            "task_id": _normalize_task_id(task_name),
            "predicted_agent": agent or "",
            "predicted_step": step or "",
            "reason": reason or "",
            "raw_response": final_response,
            "success": success
        }
    except Exception as e:
        return {
            "task_id": _normalize_task_id(task_name),
            "predicted_agent": "",
            "predicted_step": "",
            "reason": f"Error: {str(e)}",
            "raw_response": "",
            "success": False
        }


def all_at_once_parallel(client, directory_path: str, model: str, max_tokens: int, max_workers: int = 10):
    """
    Parallel version of all_at_once method

    Args:
        client: OpenAI client
        directory_path: Data directory path
        model: Model name
        max_tokens: Maximum token count
        max_workers: Maximum thread count (default 10)
    """
    print(f"\n--- Starting All-at-Once Analysis (Parallel with {max_workers} workers) ---\n")

    # Load all task data
    task_data_list = _get_task_data_list(directory_path)
    total_tasks = len(task_data_list)
    print(f"Loaded {total_tasks} tasks. Starting parallel processing...\n")

    # Collect results
    results = []

    # Execute parallel inference using thread pool
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_task = {
            executor.submit(
                _process_task_all_at_once,
                client, model, max_tokens, task_name, data
            ): task_name
            for task_name, data in task_data_list
        }

        # Display progress using tqdm
        with tqdm(total=total_tasks, desc="Processing tasks") as pbar:
            for future in as_completed(future_to_task):
                result = future.result()
                results.append(result)

                # Thread-safe output of results
                with output_lock:
                    print(f"\nPrediction for {result['task_id']}:")
                    print(result.get("raw_response", ""))
                    print("\n" + "="*50 + "\n")

                pbar.update(1)

    # Collect statistics
    successful = sum(1 for result in results if result.get("success"))
    print(f"\n--- Analysis Complete ---")
    print(f"Total tasks: {total_tasks}")
    print(f"Successful: {successful}")
    print(f"Failed: {total_tasks - successful}")
    return results


def _process_task_step_by_step(client, model, max_tokens, task_name, data):
    """
    Process single task - step_by_step method
    Note: step_by_step method itself is sequential, but different tasks can be processed in parallel
    """
    if not data:
        return {
            "task_id": _normalize_task_id(task_name),
            "predicted_agent": "",
            "predicted_step": "",
            "reason": "",
            "raw_response": "",
            "success": False
        }

    chat_history = data.get("history", [])
    problem = data.get("question", "")
    ground_truth = data.get("ground_truth", "")
    tests_status = data.get("tests_status", None)
    agent_system_intro = data.get("agent_system_intro", "")

    if not chat_history:
        return {
            "task_id": _normalize_task_id(task_name),
            "predicted_agent": "",
            "predicted_step": "",
            "reason": "",
            "raw_response": "",
            "success": False
        }

    index_agent = "name"

    output_lines = [f"--- Analyzing Task: {task_name} ---"]
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

        output_lines.append(f"Evaluating Step {idx + 1} by {agent_name}...")

        try:
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

            if not answer:
                output_lines.append("Failed to get evaluation for this step. Stopping analysis for this task.")
                error_found = True
                break

            last_answer = answer

            output_lines.append(f"LLM Evaluation: {answer}")

            if answer.lower().strip().startswith("1. yes"):
                output_lines.append(f"\nPrediction for {task_name}: Error found.")
                output_lines.append(f"Agent Name: {agent_name}")
                output_lines.append(f"Step Number: {idx + 1}")
                output_lines.append(f"Reason provided by LLM: {answer.split('Reason:', 1)[-1].strip()}")
                predicted_agent = agent_name
                predicted_step = str(idx + 1)
                reason = answer.split('Reason:', 1)[-1].strip()
                error_found = True
                break
            if answer.lower().strip().startswith("1. no"):
                output_lines.append("No significant error detected in this step.")
            else:
                output_lines.append("Warning: Unexpected response format from LLM. Continuing evaluation.")

        except Exception as e:
            output_lines.append(f"Error during evaluation: {str(e)}")
            error_found = True
            break

    if not error_found:
        output_lines.append(f"\nNo decisive errors found by step-by-step analysis in task {task_name}")

    return {
        "task_id": _normalize_task_id(task_name),
        "predicted_agent": predicted_agent,
        "predicted_step": predicted_step,
        "reason": reason,
        "raw_response": "\n".join(output_lines),
        "success": bool(predicted_agent and predicted_step)
    }


def step_by_step_parallel(client, directory_path: str, model: str, max_tokens: int, max_workers: int = 5):
    """
    Parallel version of step_by_step method
    Note: Since step_by_step itself is sequential, this only processes different tasks in parallel
    Recommended to use fewer workers (default 5) since each task has multiple API calls internally
    """
    print(f"\n--- Starting Step-by-Step Analysis (Parallel with {max_workers} workers) ---\n")

    task_data_list = _get_task_data_list(directory_path)
    total_tasks = len(task_data_list)
    print(f"Loaded {total_tasks} tasks. Starting parallel processing...\n")

    results = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_task = {
            executor.submit(
                _process_task_step_by_step,
                client, model, max_tokens, task_name, data
            ): task_name
            for task_name, data in task_data_list
        }

        with tqdm(total=total_tasks, desc="Processing tasks") as pbar:
            for future in as_completed(future_to_task):
                result = future.result()
                results.append(result)

                with output_lock:
                    print(result.get("raw_response", ""))
                    print("\n" + "="*50 + "\n")

                pbar.update(1)

    successful = sum(1 for result in results if result.get("success"))
    print(f"\n--- Analysis Complete ---")
    print(f"Total tasks: {total_tasks}")
    print(f"Errors found: {successful}")
    print(f"No errors: {total_tasks - successful}")
    return results


def _find_error_binary_search(client, model, max_tokens, chat_history, problem, answer, start, end, task_name, tests_status=None, agent_system_intro=""):
    """Recursive function for binary search to find error step"""
    if start > end:
        return (start, f"Invalid range (start={start}, end={end})")
    if start == end:
        return (start, "Error located at this step")

    index_agent = "name"

    segment_history = chat_history[start : end + 1]
    if not segment_history:
        return (start, "Empty segment")

    mid = start + (end - start) // 2

    range_description = f"from step {start + 1} to step {end + 1}"
    upper_half_desc = f"from step {start + 1} to step {mid + 1}"
    lower_half_desc = f"from step {mid + 2} to step {end + 1}"

    try:
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
            return (start, f"API call failed for segment {start}-{end}")

        result_lower = result.lower()

        if "upper half" in result_lower:
            return _find_error_binary_search(client, model, max_tokens, chat_history, problem, answer, start, mid, task_name, tests_status, agent_system_intro)
        elif "lower half" in result_lower:
            new_start = min(mid + 1, end)
            return _find_error_binary_search(client, model, max_tokens, chat_history, problem, answer, new_start, end, task_name, tests_status, agent_system_intro)
        else:
            # Random selection
            if random.randint(0, 1) == 0:
                return _find_error_binary_search(client, model, max_tokens, chat_history, problem, answer, start, mid, task_name, tests_status, agent_system_intro)
            else:
                new_start = min(mid + 1, end)
                return _find_error_binary_search(client, model, max_tokens, chat_history, problem, answer, new_start, end, task_name, tests_status, agent_system_intro)

    except Exception as e:
        return (start, f"Error during binary search: {str(e)}")


def _process_task_binary_search(client, model, max_tokens, task_name, data):
    """Process single task - binary_search method"""
    if not data:
        return {
            "task_id": _normalize_task_id(task_name),
            "predicted_agent": "",
            "predicted_step": "",
            "reason": "",
            "raw_response": "",
            "success": False
        }

    chat_history = data.get("history", [])
    problem = data.get("question", "")
    ground_truth = data.get("ground_truth", "")
    tests_status = data.get("tests_status", None)
    agent_system_intro = data.get("agent_system_intro", "")

    if not chat_history:
        return {
            "task_id": _normalize_task_id(task_name),
            "predicted_agent": "",
            "predicted_step": "",
            "reason": "",
            "raw_response": "",
            "success": False
        }

    index_agent = "name"

    try:
        # Execute binary search
        error_step, message = _find_error_binary_search(
            client, model, max_tokens, chat_history, problem, ground_truth,
            0, len(chat_history) - 1, task_name, tests_status, agent_system_intro
        )

        # Get agent info for error step
        if 0 <= error_step < len(chat_history):
            entry = chat_history[error_step]
            agent_name = entry.get(index_agent, 'Unknown Agent')

            result_text = f"Agent Name: {agent_name}\nStep Number: {error_step + 1}\nMessage: {message}"
            return {
                "task_id": _normalize_task_id(task_name),
                "predicted_agent": agent_name,
                "predicted_step": str(error_step + 1),
                "reason": message,
                "raw_response": result_text,
                "success": True
            }
        else:
            return {
                "task_id": _normalize_task_id(task_name),
                "predicted_agent": "",
                "predicted_step": "",
                "reason": f"Invalid error step: {error_step}",
                "raw_response": "",
                "success": False
            }

    except Exception as e:
        return {
            "task_id": _normalize_task_id(task_name),
            "predicted_agent": "",
            "predicted_step": "",
            "reason": f"Error: {str(e)}",
            "raw_response": "",
            "success": False
        }


def binary_search_parallel(client, directory_path: str, model: str, max_tokens: int, max_workers: int = 5):
    """
    Parallel version of binary_search method
    Recommended to use fewer workers (default 5) since each task has multiple recursive API calls internally
    """
    print(f"\n--- Starting Binary Search Analysis (Parallel with {max_workers} workers) ---\n")

    task_data_list = _get_task_data_list(directory_path)
    total_tasks = len(task_data_list)
    print(f"Loaded {total_tasks} tasks. Starting parallel processing...\n")

    results = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_task = {
            executor.submit(
                _process_task_binary_search,
                client, model, max_tokens, task_name, data
            ): task_name
            for task_name, data in task_data_list
        }

        with tqdm(total=total_tasks, desc="Processing tasks") as pbar:
            for future in as_completed(future_to_task):
                result = future.result()
                results.append(result)

                with output_lock:
                    print(f"\nPrediction for {result['task_id']}:")
                    print(result.get("raw_response", ""))
                    print("\n" + "="*50 + "\n")

                pbar.update(1)

    successful = sum(1 for result in results if result.get("success"))
    print(f"\n--- Analysis Complete ---")
    print(f"Total tasks: {total_tasks}")
    print(f"Successful: {successful}")
    print(f"Failed: {total_tasks - successful}")
    return results
