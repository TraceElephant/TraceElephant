import json
import os
import argparse
from pathlib import Path

_ACTUAL_DATA_CACHE = {}
_TOTAL_STEPS_CACHE = {}

def _to_int(value):
    try:
        return int(value)
    except Exception:
        return None

def _is_raw_test_data_format(directory_path):
    """Check if directory is in raw_test_data_ca format"""
    dir_path = Path(directory_path)
    if not dir_path.exists():
        return False
    has_runs_dirs = any('-runs-' in d.name for d in dir_path.iterdir() if d.is_dir())
    return has_runs_dirs

def _iter_runs_dirs(directory_path):
    dir_path = Path(directory_path)
    for d in dir_path.iterdir():
        if d.is_dir() and '-runs-' in d.name:
            yield d

def read_predictions(eval_file):
    if not os.path.exists(eval_file):
        print(f"Error: Evaluation file not found at {eval_file}")
        return {}

    try:
        with open(eval_file, 'r', encoding='utf-8') as file:
            lines = file.readlines()
    except Exception as e:
        print(f"Error reading evaluation file {eval_file}: {e}")
        return {}

    predictions = {}
    parsed_count = 0

    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except Exception as e:
            print(f"Warning: Could not parse JSON line in {eval_file}: {e}")
            continue

        task_id = record.get("task_id") or record.get("task_name")
        predicted_agent = record.get("predicted_agent", "")
        predicted_step = record.get("predicted_step", "")

        if task_id is None:
            print(f"Warning: Missing task_id in {eval_file} line: {line[:80]}...")
            continue

        predictions[task_id] = {
            "predicted_agent": str(predicted_agent),
            "predicted_step": str(predicted_step)
        }
        parsed_count += 1

    print(f"--- Predictions Read from {eval_file} ---")
    print(f"Successfully parsed predictions for {parsed_count} files.")
    print("=======================================")
    return predictions

def read_actual_data(data_path, task_identifier):
    """
    Read actual annotation data from data path
    Compatible with two formats:
    1. Original format: data_path/task_identifier.json
    2. raw_test_data_ca format: data_path/captain-runs-*/task_identifier/trace_metadata.json

    Returns: (mistake_agent, mistake_step, system_name, category)
    category is 'captain', 'magentic', 'swe' or 'other'
    """
    cache_key = (data_path, task_identifier)
    if cache_key in _ACTUAL_DATA_CACHE:
        return _ACTUAL_DATA_CACHE[cache_key]

    # Try original format
    labeled_json = os.path.join(data_path, f"{task_identifier}")
    if os.path.exists(labeled_json):
        try:
            with open(labeled_json, 'r', encoding='utf-8') as file:
                data = json.load(file)
            mistake_agent = data.get('mistake_agent')
            mistake_step = data.get('mistake_step')
            if mistake_agent is not None and mistake_step is not None:
                result = (str(mistake_agent), str(mistake_step), None, 'other')
                _ACTUAL_DATA_CACHE[cache_key] = result
                return result
        except Exception as e:
            print(f"Error reading {labeled_json}: {e}")

    # Try raw_test_data_ca format
    if _is_raw_test_data_format(data_path):
        dir_path = Path(data_path)
        for runs_dir in _iter_runs_dirs(dir_path):
            task_dir = runs_dir / task_identifier
            trace_metadata_file = task_dir / "trace_metadata.json"
            if trace_metadata_file.exists():
                try:
                    with open(trace_metadata_file, 'r', encoding='utf-8') as file:
                        data = json.load(file)
                    mistake_agent = data.get('mistake_agent')
                    mistake_step = data.get('mistake_step')
                    system_name = data.get('system_name')

                    # Determine category based on directory name
                    runs_dir_name = runs_dir.name.lower()
                    if runs_dir_name.startswith('captain'):
                        category = 'captain'
                    elif runs_dir_name.startswith('magentic'):
                        category = 'magentic'
                    elif runs_dir_name.startswith('swe'):
                        category = 'swe'
                    else:
                        category = 'other'

                    if mistake_agent is not None and mistake_step is not None:
                        result = (str(mistake_agent), str(mistake_step), system_name, category)
                        _ACTUAL_DATA_CACHE[cache_key] = result
                        return result
                except Exception as e:
                    print(f"Error reading {trace_metadata_file}: {e}")

    print(f"Warning: Could not find actual data for {task_identifier}")
    result = (None, None, None, None)
    _ACTUAL_DATA_CACHE[cache_key] = result
    return result

def read_total_steps(data_path, task_identifier):
    cache_key = (data_path, task_identifier)
    if cache_key in _TOTAL_STEPS_CACHE:
        return _TOTAL_STEPS_CACHE[cache_key]

    if _is_raw_test_data_format(data_path):
        dir_path = Path(data_path)
        for runs_dir in _iter_runs_dirs(dir_path):
            task_dir = runs_dir / task_identifier
            step_records_file = task_dir / "step_records.json"
            if step_records_file.exists():
                try:
                    with open(step_records_file, 'r', encoding='utf-8') as file:
                        data = json.load(file)
                    total_steps = len(data) if isinstance(data, list) else None
                    _TOTAL_STEPS_CACHE[cache_key] = total_steps
                    return total_steps
                except Exception as e:
                    print(f"Error reading {step_records_file}: {e}")

    _TOTAL_STEPS_CACHE[cache_key] = None
    return None

def evaluate_accuracy(predictions, data_path, total_files, system_total_files=None):
    correct_agent = 0
    correct_step = 0
    files_evaluated = 0
    per_system = {}

    if total_files == 0:
        print("Error: No JSON files found in the data path to evaluate against.")
        return 0.0, 0.0, {}

    print(f"\n--- Starting Evaluation ---")
    print(f"Total reference JSON files found in {data_path}: {total_files}")
    print(f"Predictions available for {len(predictions)} files.")
    print("=======================================")

    for idx, pred in predictions.items():
        # Extract task identifier from prediction key (remove .json suffix if exists)
        task_identifier = idx.replace('.json', '')

        # Use updated read_actual_data function
        actual_agent, actual_step, system_name, category = read_actual_data(data_path, task_identifier)

        if actual_agent is not None and actual_step is not None:
            files_evaluated += 1
            agent_correct = actual_agent in pred['predicted_agent']
            step_correct = actual_step in pred['predicted_step']

            if agent_correct:
                correct_agent += 1
            if step_correct:
                correct_step += 1

            # Statistics by system_name
            if system_name is None:
                system_name = "unknown"
            if system_name not in per_system:
                per_system[system_name] = {
                    "correct_agent": 0,
                    "correct_step": 0,
                    "files_evaluated": 0,
                }
            per_system[system_name]["files_evaluated"] += 1
            if agent_correct:
                per_system[system_name]["correct_agent"] += 1
            if step_correct:
                per_system[system_name]["correct_step"] += 1

        else:
            print(f"Skipping evaluation for {idx} due to issues reading actual data.")

    print("\n--- Evaluation Summary ---")
    print(f"Total reference files in data_path: {total_files}")
    print(f"Predictions parsed from eval file:  {len(predictions)}")
    print(f"Files evaluated (prediction found & actual data read): {files_evaluated}")
    print(f"Correct Agent Predictions: {correct_agent}")
    print(f"Correct Step Predictions:  {correct_step}")

    if per_system:
        print("\n--- Per-System Summary ---")
        for system_name, stats in sorted(per_system.items()):
            # Use actual evaluated file count as denominator
            denom = stats["files_evaluated"]
            agent_acc = (stats["correct_agent"] / denom) * 100 if denom > 0 else 0.0
            step_acc = (stats["correct_step"] / denom) * 100 if denom > 0 else 0.0
            if system_total_files and system_name in system_total_files:
                total = system_total_files[system_name]
                print(
                    f"{system_name}: Agent {agent_acc:.2f}%, Step {step_acc:.2f}% "
                    f"(evaluated {denom}/{total})"
                )
            else:
                print(
                    f"{system_name}: Agent {agent_acc:.2f}%, Step {step_acc:.2f}% "
                    f"(evaluated {denom})"
                )

    # Use actual evaluated file count as denominator, not total file count
    agent_accuracy = (correct_agent / files_evaluated) * 100 if files_evaluated > 0 else 0.0
    step_accuracy = (correct_step / files_evaluated) * 100 if files_evaluated > 0 else 0.0

    return (
        agent_accuracy,
        step_accuracy,
    )

def main():
    parser = argparse.ArgumentParser(description="Evaluate agent and step prediction accuracy from a JSONL prediction file.")
    parser.add_argument(
        "--data_path",
        type=str,
        default='../Who&When/Algorithm-Generated',
        help="Path to the directory containing the ground truth files."
    )
    parser.add_argument(
        "--eval_file",
        type=str,
        required=True,
        help="Path to the evaluation log file containing the predictions."
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="",
        help="Optional path to write evaluation summary."
    )
    args = parser.parse_args()
    
    data_path = args.data_path
    eval_file = args.eval_file

    if not os.path.isdir(data_path):
        print(f"Error: Data directory not found at {data_path}")
        actual_total_files = 0
        system_total_files = None
    else:
        try:
            # Check data format and count files
            if _is_raw_test_data_format(data_path):
                # raw_test_data_ca format: count task directories
                dir_path = Path(data_path)
                actual_total_files = 0
                system_total_files = {}
                for runs_dir in _iter_runs_dirs(dir_path):
                    if runs_dir.is_dir():
                        task_dirs = [d for d in runs_dir.iterdir() if d.is_dir() and (d / "trace_metadata.json").exists()]
                        for task_dir in task_dirs:
                            actual_total_files += 1
                            trace_metadata_file = task_dir / "trace_metadata.json"
                            try:
                                with open(trace_metadata_file, 'r', encoding='utf-8') as file:
                                    data = json.load(file)
                                system_name = data.get("system_name") or "unknown"
                            except Exception:
                                system_name = "unknown"
                            system_total_files[system_name] = system_total_files.get(system_name, 0) + 1
                print(f"Detected raw_test_data_ca format: {actual_total_files} tasks found")
            else:
                # Original format: count JSON files
                json_files_in_data_path = [
                    f for f in os.listdir(data_path)
                    if f.endswith('.json') and os.path.isfile(os.path.join(data_path, f))
                ]
                actual_total_files = len(json_files_in_data_path)
                system_total_files = None
                print(f"Detected original JSON format: {actual_total_files} files found")
        except Exception as e:
            print(f"Error reading data directory {data_path}: {e}")
            actual_total_files = 0
            system_total_files = None

    predictions = read_predictions(eval_file)

    # Calculate actual number of evaluated files
    files_evaluated = 0
    for idx, pred in predictions.items():
        task_identifier = idx.replace('.json', '')
        actual_agent, actual_step, system_name, category = read_actual_data(data_path, task_identifier)
        if actual_agent is not None and actual_step is not None:
            files_evaluated += 1

    (
        agent_accuracy,
        step_accuracy,
    ) = evaluate_accuracy(
        predictions,
        data_path,
        actual_total_files,
        system_total_files=system_total_files,
    )

    print("\n--- Final Accuracy Results ---")
    print(f"Evaluation File: {eval_file}")
    print(f"Data Path:       {data_path}")
    print(f"Agent Accuracy: {agent_accuracy:.2f}%")
    print(f"Step Accuracy:  {step_accuracy:.2f}%")
    print(f"(Accuracy calculated based on {files_evaluated} evaluated files out of {actual_total_files} total files in data path)")

    if args.output_file:
        try:
            with open(args.output_file, 'w', encoding='utf-8') as out_f:
                out_f.write("--- Final Accuracy Results ---\n")
                out_f.write(f"Evaluation File: {eval_file}\n")
                out_f.write(f"Data Path:       {data_path}\n")
                out_f.write(f"Agent Accuracy: {agent_accuracy:.2f}%\n")
                out_f.write(f"Step Accuracy:  {step_accuracy:.2f}%\n")
                out_f.write(f"(Accuracy calculated based on {files_evaluated} evaluated files out of {actual_total_files} total files in data path)\n")

            print(f"Saved evaluation summary to {args.output_file}")
        except Exception as e:
            print(f"Error writing output_file {args.output_file}: {e}")

if __name__ == "__main__":
    main()
