#
# Run this file to download the human_eval dataset, and create a corresponding testbed scenario:
# (default: ../scenarios/human_eval_two_agents_gpt4.jsonl and ./scenarios/human_eval_two_agents_gpt35.jsonl)
#

import requests
import tarfile
import json
import os
import re
import argparse
from autogen.agentchat.contrib.agent_builder import AgentBuilder

URL = "https://people.eecs.berkeley.edu/~hendrycks/MATH.tar"

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
    'MATH/test/algebra/2130.json',
    'MATH/test/algebra/1659.json',
    'MATH/test/algebra/1209.json',
    'MATH/test/algebra/816.json',
    'MATH/test/algebra/1570.json',
    'MATH/test/algebra/1065.json',
    'MATH/test/algebra/400.json',
    'MATH/test/algebra/2649.json',
    'MATH/test/algebra/2167.json',
    'MATH/test/algebra/791.json',
    'MATH/test/algebra/2022.json',
    'MATH/test/algebra/1527.json',
    'MATH/test/algebra/512.json',
    'MATH/test/algebra/142.json',
    'MATH/test/algebra/841.json',
    'MATH/test/algebra/904.json',
    'MATH/test/algebra/1462.json',
    'MATH/test/algebra/1032.json',
    'MATH/test/algebra/338.json',
    'MATH/test/algebra/1248.json',
    'MATH/test/algebra/1862.json',
    'MATH/test/algebra/441.json',
    'MATH/test/algebra/2576.json',
    'MATH/test/algebra/2126.json',
    'MATH/test/algebra/2433.json',
    'MATH/test/algebra/945.json',
    'MATH/test/algebra/416.json',
    'MATH/test/algebra/1423.json',
    'MATH/test/algebra/1264.json',
    'MATH/test/algebra/251.json',
    'MATH/test/algebra/1634.json',
    'MATH/test/algebra/314.json',
    'MATH/test/algebra/744.json',
    'MATH/test/algebra/2274.json',
    'MATH/test/algebra/1458.json',
    'MATH/test/algebra/528.json',
    'MATH/test/algebra/482.json',
    'MATH/test/algebra/178.json',
    'MATH/test/algebra/2331.json',
    'MATH/test/algebra/206.json',
    'MATH/test/algebra/1233.json',
    'MATH/test/algebra/713.json',
    'MATH/test/algebra/1376.json',
    'MATH/test/algebra/969.json',
    'MATH/test/algebra/2736.json',
    'MATH/test/algebra/986.json',
    'MATH/test/algebra/77.json',
    'MATH/test/geometry/379.json',
    'MATH/test/geometry/396.json',
    'MATH/test/geometry/115.json',
    'MATH/test/geometry/1120.json',
    'MATH/test/geometry/1065.json',
    'MATH/test/geometry/953.json',
    'MATH/test/geometry/1032.json',
    'MATH/test/geometry/338.json',
    'MATH/test/geometry/787.json',
    'MATH/test/geometry/154.json',
    'MATH/test/geometry/504.json',
    'MATH/test/geometry/912.json',
    'MATH/test/geometry/380.json',
    'MATH/test/geometry/103.json',
    'MATH/test/geometry/800.json',
    'MATH/test/geometry/197.json',
    'MATH/test/geometry/528.json',
    'MATH/test/geometry/178.json',
    'MATH/test/geometry/656.json',
    'MATH/test/prealgebra/379.json',
    'MATH/test/prealgebra/2075.json',
    'MATH/test/prealgebra/1209.json',
    'MATH/test/prealgebra/1823.json',
    'MATH/test/prealgebra/1570.json',
    'MATH/test/prealgebra/1065.json',
    'MATH/test/prealgebra/953.json',
    'MATH/test/prealgebra/2022.json',
    'MATH/test/prealgebra/1874.json',
    'MATH/test/prealgebra/841.json',
    'MATH/test/prealgebra/904.json',
    'MATH/test/prealgebra/1198.json',
    'MATH/test/prealgebra/1462.json',
    'MATH/test/prealgebra/1032.json',
    'MATH/test/prealgebra/768.json',
    'MATH/test/prealgebra/2034.json',
    'MATH/test/prealgebra/857.json',
    'MATH/test/prealgebra/154.json',
    'MATH/test/prealgebra/1531.json',
    'MATH/test/prealgebra/1024.json',
    'MATH/test/prealgebra/1474.json',
    'MATH/test/prealgebra/380.json',
    'MATH/test/prealgebra/2063.json',
    'MATH/test/prealgebra/1566.json',
    'MATH/test/prealgebra/1136.json',
    'MATH/test/prealgebra/945.json',
    'MATH/test/prealgebra/1970.json',
    'MATH/test/prealgebra/1423.json',
    'MATH/test/prealgebra/1073.json',
    'MATH/test/prealgebra/1634.json',
    'MATH/test/prealgebra/2018.json',
    'MATH/test/prealgebra/1321.json',
    'MATH/test/prealgebra/1458.json',
    'MATH/test/prealgebra/1008.json',
    'MATH/test/precalculus/683.json',
    'MATH/test/precalculus/396.json',
    'MATH/test/precalculus/1120.json',
    'MATH/test/precalculus/545.json',
    'MATH/test/precalculus/1065.json',
    'MATH/test/precalculus/791.json',
    'MATH/test/precalculus/904.json',
    'MATH/test/precalculus/1032.json',
    'MATH/test/precalculus/768.json',
    'MATH/test/precalculus/1248.json',
    'MATH/test/precalculus/857.json',
    'MATH/test/precalculus/1161.json',
    'MATH/test/precalculus/504.json',
    'MATH/test/precalculus/1024.json',
    'MATH/test/precalculus/441.json',
    'MATH/test/precalculus/912.json',
    'MATH/test/precalculus/695.json',
    'MATH/test/precalculus/553.json',
    'MATH/test/precalculus/1136.json',
    'MATH/test/precalculus/103.json',
    'MATH/test/precalculus/945.json',
    'MATH/test/number_theory/729.json',
    'MATH/test/number_theory/1120.json',
    'MATH/test/number_theory/545.json',
    'MATH/test/number_theory/1065.json',
    'MATH/test/number_theory/284.json',
    'MATH/test/number_theory/457.json',
    'MATH/test/number_theory/1032.json',
    'MATH/test/number_theory/338.json',
    'MATH/test/number_theory/787.json',
    'MATH/test/number_theory/1248.json',
    'MATH/test/number_theory/154.json',
    'MATH/test/number_theory/1161.json',
    'MATH/test/number_theory/504.json',
    'MATH/test/number_theory/1024.json',
    'MATH/test/number_theory/441.json',
    'MATH/test/number_theory/695.json',
    'MATH/test/number_theory/380.json',
    'MATH/test/number_theory/1136.json',
    'MATH/test/number_theory/103.json',
    'MATH/test/number_theory/601.json',
    'MATH/test/number_theory/314.json',
    'MATH/test/intermediate_algebra/729.json',
    'MATH/test/intermediate_algebra/379.json',
    'MATH/test/intermediate_algebra/2130.json',
    'MATH/test/intermediate_algebra/1823.json',
    'MATH/test/intermediate_algebra/1570.json',
    'MATH/test/intermediate_algebra/2167.json',
    'MATH/test/intermediate_algebra/2188.json',
    'MATH/test/intermediate_algebra/2022.json',
    'MATH/test/intermediate_algebra/142.json',
    'MATH/test/intermediate_algebra/1874.json',
    'MATH/test/intermediate_algebra/841.json',
    'MATH/test/intermediate_algebra/904.json',
    'MATH/test/intermediate_algebra/1198.json',
    'MATH/test/intermediate_algebra/1462.json',
    'MATH/test/intermediate_algebra/1032.json',
    'MATH/test/intermediate_algebra/2171.json',
    'MATH/test/intermediate_algebra/154.json',
    'MATH/test/intermediate_algebra/1024.json',
    'MATH/test/intermediate_algebra/1474.json',
    'MATH/test/intermediate_algebra/441.json',
    'MATH/test/intermediate_algebra/2126.json',
    'MATH/test/intermediate_algebra/1566.json',
    'MATH/test/intermediate_algebra/1136.json',
    'MATH/test/intermediate_algebra/800.json',
    'MATH/test/intermediate_algebra/945.json',
    'MATH/test/intermediate_algebra/416.json',
    'MATH/test/intermediate_algebra/1423.json',
    'MATH/test/intermediate_algebra/1073.json',
    'MATH/test/intermediate_algebra/251.json',
    'MATH/test/intermediate_algebra/1634.json',
    'MATH/test/intermediate_algebra/601.json',
    'MATH/test/intermediate_algebra/2018.json',
    'MATH/test/intermediate_algebra/1321.json',
    'MATH/test/intermediate_algebra/894.json',
    'MATH/test/intermediate_algebra/1458.json',
    'MATH/test/intermediate_algebra/1008.json',
    'MATH/test/counting_and_probability/396.json',
    'MATH/test/counting_and_probability/816.json',
    'MATH/test/counting_and_probability/115.json',
    'MATH/test/counting_and_probability/1120.json',
    'MATH/test/counting_and_probability/545.json',
    'MATH/test/counting_and_probability/1065.json',
    'MATH/test/counting_and_probability/400.json',
    'MATH/test/counting_and_probability/512.json',
    'MATH/test/counting_and_probability/904.json',
    'MATH/test/counting_and_probability/292.json',
    'MATH/test/counting_and_probability/857.json',
    'MATH/test/counting_and_probability/504.json',
    'MATH/test/counting_and_probability/695.json',
    'MATH/test/counting_and_probability/103.json',
    'MATH/test/counting_and_probability/416.json',
    'MATH/test/counting_and_probability/894.json',
    'MATH/test/counting_and_probability/528.json',
    'MATH/test/counting_and_probability/482.json'
]

SELECTED_PROBLEMS_REDUCED = [
    "MATH/test/algebra/2144.json",
    "MATH/test/algebra/1997.json",
    "MATH/test/algebra/2072.json",
    "MATH/test/algebra/2137.json",
    "MATH/test/algebra/2557.json",
    "MATH/test/algebra/2045.json",
    "MATH/test/algebra/2499.json",
    "MATH/test/counting_and_probability/483.json",
    "MATH/test/intermediate_algebra/590.json",
    "MATH/test/prealgebra/1511.json",
    "MATH/test/intermediate_algebra/935.json",
    "MATH/test/prealgebra/808.json",
    "MATH/test/number_theory/233.json",
    "MATH/test/number_theory/960.json",
    "MATH/test/precalculus/551.json",
    "MATH/test/counting_and_probability/909.json",
    "MATH/test/algebra/2417.json",
]


def download_math():
    """Download the MATH dataset (if not already downloaded).
    Return a JSON dictionary of selected problems."""

    selected_problems = dict()
    reduced_selected_problems = dict()

    if not os.path.isdir(DOWNLOADS_DIR):
        os.mkdir(DOWNLOADS_DIR)

    tar_file = os.path.join(DOWNLOADS_DIR, "MATH.tar")

    if not os.path.isfile(tar_file):
        # Send a HTTP request to the URL
        response = requests.get(URL, stream=True)
        response.raise_for_status()

        # If the HTTP request returns a status code 200, proceed
        with open(tar_file, "wb") as fh:
            for chunk in response.iter_content(chunk_size=512):
                fh.write(chunk)

    # Extract selected problems
    tar = tarfile.open(tar_file)
    for member in tar.getmembers():
        if member.name in SELECTED_PROBLEMS:
        # if ".json" in member.name and "test" in member.name:
            # print(f"Extracting: {member.name}")
            content = tar.extractfile(member).read()
            selected_problems[member.name] = json.loads(content)
        if member.name in SELECTED_PROBLEMS_REDUCED:
            content = tar.extractfile(member).read()
            reduced_selected_problems[member.name] = json.loads(content)

    return selected_problems, reduced_selected_problems


def create_jsonl(name, problems, template, agent_list = None, config_list = "OAI_CONFIG_LIST", config_list2="OAI_CONFIG_LIST"):
    """Creates a JSONL scenario file with a given name, dictionary of MATH problems, and template path."""

    # Create a task directory if it doesn't exist
    if not os.path.isdir(TASKS_DIR):
        os.mkdir(TASKS_DIR)

    # Create the jsonl file
    if "OAI_CONFIG_LIST" not in config_list:
        config_list_name = ""
    else:
        config_list_name = config_list
    with open(os.path.join(TASKS_DIR, f"{name}{config_list.replace('OAI_CONFIG_LIST', '')}.jsonl"), "wt") as fh:
        for item in problems.items():
            data = item[1]

            task_id = item[0].replace("MATH/", "").replace(".json", "").replace("/", "_")
            # print(f"Converting: [{item[0]}] {task_id}")

            record = {
                "id": task_id,
                "template": os.path.join(os.path.pardir, template),
                "substitutions": {
                    "prompt.txt": {"__PROMPT__": data["problem"]},
                    "expected_answer.txt": {"__ANSWER__": data["solution"]},
                    "agent_list.txt": {"__AGENT_LIST__": json.dumps(agent_list)},
                    "scenario.py": {
                        "__CONFIG_LIST_PATH__": config_list,
                        "__CONFIG_LIST_PATH2__": config_list2,
                        "__AGENT_SAVE_PATH__": SAVE_DIR,
                        "__LIBRARY_PATH__": f"{BENCH_DIR}/agent_library.json",
                        "__TOOL_CORPUS__": f"{AG_PATH}/tools/tool_description.tsv",
                        "__TOOL_ROOT__": f"{AG_PATH}/tools"
                    }
                },
            }
            fh.write(json.dumps(record).strip() + "\n")


###############################################################################
def main(args):
    problems, reduced_problems = download_math()
    building_task = """We need a group of math experts to solve some math problems. 
Those problems are in the fields of algebra, counting and probability, geometry, intermediate algebra, number theory, pre-algebra, and pre-calculus.
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
    builder = AgentBuilder(config_file_or_env=args.config_list,
                        builder_model_tags=['gpt-4', '1106', '0125', 'claude3', 'haiku', 'sonnet', 'gemini-1.5', 'llama3', '8b', '70b', 'mixtral', '8x22b', '8x7b'],
                        agent_model_tags=['gpt-4', '1106', '0125', 'claude3', 'haiku', 'sonnet', 'gemini-1.5', 'llama3', '8b', '70b', 'mixtral', '8x22b', '8x7b'],
                        max_agents=10)
    _, agent_configs = builder.build(building_task, default_llm_config, coding=True)
    builder.save(f"{SAVE_DIR}/autobuild.json")

    for t in templates.items():
        create_jsonl(f"math_{t[0]}", problems, t[1], agent_list=agent_configs, config_list=args.config_list, config_list2=args.config_list2)
    
    for t in templates.items():
        create_jsonl(f"r_math_{t[0]}", reduced_problems, t[1], agent_list=agent_configs, config_list=args.config_list, config_list2=args.config_list2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config-list', type=str, default="OAI_CONFIG_LIST")
    parser.add_argument('--config-list2', type=str, default="OAI_CONFIG_LIST")
    args = parser.parse_args()
    main(args)
