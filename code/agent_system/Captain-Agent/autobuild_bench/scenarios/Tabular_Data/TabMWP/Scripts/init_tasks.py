#
# Run this file to download the human_eval dataset, and create a corresponding testbed scenario:
# (default: ../scenarios/human_eval_two_agents_gpt4.jsonl and ./scenarios/human_eval_two_agents_gpt35.jsonl)
#

import requests
import tarfile
import json
import os
import re
from autogen.agentchat.contrib.agent_builder import AgentBuilder

SCRIPT_PATH = os.path.realpath(__file__)
SCRIPT_NAME = os.path.basename(SCRIPT_PATH)
SCRIPT_DIR = os.path.dirname(SCRIPT_PATH)

SCENARIO_DIR = os.path.realpath(os.path.join(SCRIPT_DIR, os.path.pardir))
TEMPLATES_DIR = os.path.join(SCENARIO_DIR, "Templates")
TASKS_DIR = os.path.join(SCENARIO_DIR, "Tasks")
DOWNLOADS_DIR = os.path.join(SCENARIO_DIR, "Downloads")
SAVE_DIR = os.path.join(SCENARIO_DIR, "Saved_agents")

with open('../Downloads/problems_dev_sample.json', 'r') as file:
    DATA = json.load(file)

SELECTED_PROBLEMS = [
    '19',
    '589',
    '617',
    '33',
    '139',
    '150',
    '10',
    '35',
    '37',
    '2',
    '3',
    '11',
    '9',
    '18',
    '50'
]


def create_jsonl(name, problems, template, agent_list=None):
    """Creates a JSONL scenario file with a given name, dictionary of MATH problems, and template path."""

    # Create a task directory if it doesn't exist
    if not os.path.isdir(TASKS_DIR):
        os.mkdir(TASKS_DIR)

    # Create the jsonl file
    with open(os.path.join(TASKS_DIR, name + ".jsonl"), "wt") as fh:
        for item in problems.items():
            data = item[1]
            task_id = item[0]
            # delete the follow row
            print(f"Converting: [{item[0]}] {task_id}")
            if data["choices"] is None:
                prompt_tmp = data["question"]
            else:
                prompt_tmp = data["question"] + '\nCHOICES:' + str(data["choices"])
            record = {
                "id": task_id,
                "template": os.path.join(os.path.pardir, template),
                "substitutions": {
                    "prompt.txt": {"__PROMPT__": prompt_tmp},
                    "expected_answer.txt": {"__ANSWER__": data["answer"]},
                    "table.txt": {"__TABLE__": f"Table Title: {data['table_title']}\nTable Content:\n{data['table']}"},
                    "agent_list.txt": {"__AGENT_LIST__": json.dumps(agent_list)},
                },
            }

            fh.write(json.dumps(record).strip() + "\n")


def selected_problems():
    select_problems = dict()
    for item in DATA.items():
        if item[0] in SELECTED_PROBLEMS:
            print(f"Extracting: No.{item[0]}")
            select_problems[item[0]] = item[1]
    return select_problems


###############################################################################
def main():
    problems = selected_problems()
    building_task = """We need a group of experts to solve some problems with the help of tabular data.
There are two kinds of problems: free-text problem with a numerical answer, multi-choice problem with a textual answer. 
To reach the right answer, some simple math calculations are required.
They need to determine the type of problem, and use the questions and the given tabular data to reach the right answer.
They need to solve the problem collaboratively and check each other's answer. Also, they can write python code themselves to help solving the task if needed.
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
    builder = AgentBuilder(config_file_or_env='OAI_CONFIG_LIST',
                           builder_model='gpt-4-1106',
                           agent_model='gpt-4-1106',
                           max_agents=10)
    _, agent_configs = builder.build(building_task, default_llm_config, coding=True)
    builder.save(f"{SAVE_DIR}/autobuild.json")
    
    for t in templates.items():
        create_jsonl(f"question_{t[0]}", problems, t[1], agent_list=agent_configs)


if __name__ == "__main__" and __package__ is None:
    main()
