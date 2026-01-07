#
# Run this file to download the human_eval dataset, and create a corresponding testbed scenario:
# (default: ../scenarios/human_eval_two_agents_gpt4.jsonl and ./scenarios/human_eval_two_agents_gpt35.jsonl)
#
import json
import os
import re
import requests
import argparse
from bs4 import BeautifulSoup
from autogen.agentchat.contrib.agent_builder import AgentBuilder


SCRIPT_PATH = os.path.realpath(__file__)
SCRIPT_NAME = os.path.basename(SCRIPT_PATH)
SCRIPT_DIR = os.path.dirname(SCRIPT_PATH)

SCENARIO_DIR = os.path.realpath(os.path.join(SCRIPT_DIR, os.path.pardir))
BENCH_DIR = os.path.realpath(os.path.join(os.path.join(SCENARIO_DIR, os.path.pardir), os.path.pardir))
AG_PATH = os.path.realpath(os.path.join(os.path.join(BENCH_DIR, os.path.pardir), os.path.pardir))
TEMPLATES_DIR = os.path.join(SCENARIO_DIR, "Templates")
TASKS_DIR = os.path.join(SCENARIO_DIR, "Tasks")
DOWNLOADS_DIR = os.path.join(SCENARIO_DIR, "Downloads")
SAVE_DIR = os.path.join(SCENARIO_DIR, "Saved_agents")

SELECTED_PROBLEMS = [
    "gnn_1",
    "gnn_2",
    "gnn_3",
    "text_7",
    "text_8",
    "text_9",
    "molecular_15",
    "molecular_16",
    "molecular_17",
    "image_22",
    "image_23",
    "image_24",
    "video_45",
    "video_46",
    "video_47",
    "video_48",
    "time_series_60",
    "time_series_61",
    "time_series_62",
    "time_series_63",
    "time_series_64"
]

def get_readme(url):
    response = requests.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')
        return soup.find("article").text
    else:
        print(f"Failed to retrieve the page. Status code: {response.status_code}")
        return False


def create_jsonl(name, dataset, template, agent_list=None, readme_cache=None, config_list = "OAI_CONFIG_LIST", config_list2="OAI_CONFIG_LIST"):
    """Creates a JSONL scenario file with a given name, dictionary of MATH problems, and template path."""

    # Create a task directory if it doesn't exist
    if not os.path.isdir(TASKS_DIR):
        os.mkdir(TASKS_DIR)

    # Create the jsonl file
    file_path = os.path.join(TASKS_DIR, f"{name}{config_list.replace('OAI_CONFIG_LIST', '')}.jsonl")
    print(f"Current Path: {file_path}")
    with open(file_path, "wt") as fh:
        for item in dataset:
            data = json.loads(item)

            domain = data['domain'].lower().replace(' ', '_').replace('-', '_')
            task_id = f"{domain}_{data['id']}"
            if task_id not in SELECTED_PROBLEMS:
                continue

            print(f"Converting: {task_id}")

            # readme = readme_cache.get(task_id, None)
            # if readme is None:
            #     readme = get_readme(data['readme'])
            #     readme_cache[task_id] = readme

            record = {
                "id": task_id,
                "template": os.path.join(os.path.pardir, template),
                "substitutions": {
                    "prompt.txt": {"__PROMPT__": data["instruction"]},
                    "readme.txt": {"__README__": data["oracle_segmet"]},
                    "expected_answer.txt": {"__ANSWER__": json.dumps(data["arguments"])},
                    "agent_list.txt": {"__AGENT_LIST__": json.dumps(agent_list)},
                    "scenario.py": {
                        "__AGENT_SAVE_PATH__": SAVE_DIR,
                        "__LIBRARY_PATH__": f"{BENCH_DIR}/agent_library.json",
                        "__TOOL_CORPUS__": f"{AG_PATH}/tools/tool_description.tsv",
                        "__TOOL_ROOT__": f"{AG_PATH}/tools",
                        "__CONFIG_LIST_PATH__": config_list,
                        "__CONFIG_LIST_PATH2__": config_list2
                    }
                },
            }

            fh.write(json.dumps(record).strip() + "\n")


###############################################################################
def main(args):
    with open(f"{DOWNLOADS_DIR}/ML_Bench_quarter.jsonl") as f:
        dataset = f.readlines()

    building_task = """We need a group of machine learning developers to satisfied the user's instruction. 
Those problems are in the fields of GNN, text, molecular, image, multimodal, video, time-series, and attention usage.
These experts will be given a user's instruction and a readme file. 
Their goal is to write a python bash script to fulfill user's need, taking care of the arguments in the script to match the user's instruction.
"""

    # list all directories in the Templates directory
    # and populate a dictionary with the name and path
    templates = {}
    for entry in os.scandir(TEMPLATES_DIR):
        if entry.is_dir():
            templates[re.sub(r"\s", "", entry.name)] = entry.path

    default_llm_config = {
        "temperature": 1,
        "top_p": 0.95,
        "max_tokens": 1024,
    }

    ## build agents
    builder = AgentBuilder(config_file_or_env=args.config_list,
                        builder_model_tags=['gpt-4', '1106', '0125', 'claude3', 'haiku', 'sonnet', 'gemini-1.5', 'llama3', '8b', '70b', 'mixtral', '8x22b', '8x7b'],
                        agent_model_tags=['gpt-4', '1106', '0125', 'claude3', 'haiku', 'sonnet', 'gemini-1.5', 'llama3', '8b', '70b', 'mixtral', '8x22b', '8x7b'],
                        max_agents=10)
    _, agent_configs = builder.build(building_task, default_llm_config, coding=True)
    builder.save(f"{SAVE_DIR}/autobuild.json")

    readme_cache = {}
    for t in templates.items():
        create_jsonl(f"ml_bench_{t[0]}",
                     dataset,
                     t[1],
                     agent_list=agent_configs,
                     readme_cache=readme_cache,
                     config_list=args.config_list,
                     config_list2=args.config_list2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config-list', type=str, default="OAI_CONFIG_LIST_4omini")
    parser.add_argument('--config-list2', type=str, default="OAI_CONFIG_LIST_4omini")
    args = parser.parse_args()
    main(args)

