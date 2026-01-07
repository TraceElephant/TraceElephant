#
# Run this file to download the human_eval dataset, and create a corresponding testbed scenario:
# (default: ../scenarios/human_eval_two_agents_gpt4.jsonl and ./scenarios/human_eval_two_agents_gpt35.jsonl)
#

import argparse
import requests
import gzip
import io
import json
import os
import base64
import re
from autogen.agentchat.contrib.agent_builder import AgentBuilder

URL = "https://github.com/openai/human-eval/raw/master/data/HumanEval.jsonl.gz"

SCRIPT_PATH = os.path.realpath(__file__)
SCRIPT_NAME = os.path.basename(SCRIPT_PATH)
SCRIPT_DIR = os.path.dirname(SCRIPT_PATH)

SCENARIO_DIR = os.path.realpath(os.path.join(SCRIPT_DIR, os.path.pardir))
TEMPLATES_DIR = os.path.join(SCENARIO_DIR, "Templates")
TASKS_DIR = os.path.join(SCENARIO_DIR, "Tasks")
DOWNLOADS_DIR = os.path.join(SCENARIO_DIR, "Downloads")
SAVE_DIR = os.path.join(SCENARIO_DIR, "Saved_agents")

# A selected subset of HumanEval problems to work with during development
REDUCED_SET = [
    "HumanEval/2",
    "HumanEval/26",
    "HumanEval/32",
    "HumanEval/36",
    "HumanEval/38",
    "HumanEval/41",
    "HumanEval/50",
    "HumanEval/56",
    "HumanEval/65",
    "HumanEval/67",
    "HumanEval/84",
    "HumanEval/85",
    "HumanEval/86",
    "HumanEval/89",
    "HumanEval/99",
    "HumanEval/104",
    "HumanEval/113",
    "HumanEval/115",
    "HumanEval/120",
    "HumanEval/124",
    "HumanEval/126",
    "HumanEval/132",
    "HumanEval/135",
    "HumanEval/140",
    "HumanEval/146",
]


def download_human_eval():
    """Download the HumanEval dataset, un-gzips it, and returns a list of its parsed JSON objects."""

    # Send a HTTP request to the URL of the file
    response = requests.get(URL)

    # Ensure we raise an error if the download failed
    response.raise_for_status()

    # Create a BytesIO object from the response content
    buffer = io.BytesIO(response.content)

    # Read the file, line by line, populating a list of parsed JSON objects
    results = []
    with gzip.GzipFile(fileobj=buffer) as f_in:
        for line in f_in:
            # Parse each line as JSON
            results.append(json.loads(line))

    return results


def create_jsonl(name, tasks, template, agent_list = None, config_list="OAI_CONFIG_LIST", config_list2="OAI_CONFIG_LIST"):
    """Creates a JSONL scenario file with a given name, list of HumanEval tasks, and template path."""

    # Create a task directory if it doesn't exist
    task_dir = os.path.join(SCENARIO_DIR, "Tasks")
    if not os.path.isdir(task_dir):
        os.mkdir(task_dir)

    # Create the jsonl file
    with open(os.path.join(TASKS_DIR, f"{name}{config_list.replace('OAI_CONFIG_LIST', '')}.jsonl"), "wt") as fh:
        for task in tasks:
            print(f"Converting: [{name}] {task['task_id']}")

            record = {
                "id": task["task_id"].replace("/", "_"),
                "template": os.path.join(os.path.pardir, template),
                "substitutions": {
                    "scenario.py": {
                        "__ENTRY_POINT__": task["entry_point"],
                        "__SELECTION_METHOD__": "auto",
                        "__AGENT_SAVE_PATH__": SAVE_DIR,
                        "__CONFIG_LIST_PATH__": config_list,
                        "__CONFIG_LIST_PATH2__": config_list2
                    },
                    "prompt.txt": {"__PROMPT__": task["prompt"]},
                    "coding/my_tests.py": {
                        "__TEST__": task["test"],
                        "__PROMPT__": task["prompt"]
                    },
                    "agent_list.txt": {"__AGENT_LIST__": json.dumps(agent_list)}
                },
            }

            fh.write(json.dumps(record).strip() + "\n")


###############################################################################
def main(args):
    human_eval = download_human_eval()
    reduced_human_eval = [t for t in human_eval if t["task_id"] in REDUCED_SET]

    building_task = """We need a group of programming experts to tackle a variety of complex tasks. 
These tasks involve understanding function signatures, docstrings, and bodies.
They need to solve the problem collaboratively and check each other's answer. Also, they can write python code themselves to help solving the task if needed.
"""

    default_llm_config = {
        "temperature": 1,
        "top_p": 0.95,
        "max_tokens": 1024,
    }

    code_execution_config = {
        "last_n_messages": 2,
        "work_dir": "coding",
        "use_docker": False,
        "timeout": 10,
    }

    # list all directories in the Templates directory
    # and populate a dictionary with the name and path
    templates = {}
    for entry in os.scandir(TEMPLATES_DIR):
        if entry.is_dir():
            templates[re.sub(r"\s", "", entry.name)] = entry.path

    # build agents
    builder = AgentBuilder(config_file_or_env=args.config_list,
                        builder_model_tags=['gpt-4', '1106', '0125', 'claude3', 'haiku', 'sonnet', 'gemini-1.5', 'llama3', '8b', '70b', 'mixtral', '8x22b', '8x7b'],
                        agent_model_tags=['gpt-4', '1106', '0125', 'claude3', 'haiku', 'sonnet', 'gemini-1.5', 'llama3', '8b', '70b', 'mixtral', '8x22b', '8x7b'],
                        max_agents=10)
    _, agent_configs = builder.build(building_task,
                                     default_llm_config,
                                     code_execution_config=code_execution_config,
                                     coding=True)
    builder.save(f"{SAVE_DIR}/autobuild.json")

    # Create the various combinations of [models] x [templates]
    for t in templates.items():
        create_jsonl(f"human_eval_{t[0]}", human_eval, t[1], agent_list=agent_configs, config_list=args.config_list, config_list2=args.config_list2)
        create_jsonl(f"r_human_eval_{t[0]}", reduced_human_eval, t[1], agent_list=agent_configs, config_list=args.config_list, config_list2=args.config_list2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-list", type=str, default="OAI_CONFIG_LIST")
    parser.add_argument("--config-list2", type=str, default="OAI_CONFIG_LIST")
    args = parser.parse_args()
    main(args)

