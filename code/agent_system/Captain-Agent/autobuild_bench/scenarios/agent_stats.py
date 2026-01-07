import os
import json
from matplotlib import pyplot as plt
from pathlib import Path


def find_files(directory, file_name):
    path = Path(directory)
    return list(path.rglob(file_name))

if __name__ == "__main__":
    directory = '/linxindisk/linxin/llm/autogen-autobuild-dev/autobuild_bench/scenarios/math/MATH/Results/math_MetaAgent_0125'
    file_name = 'console_log.txt'
    files = find_files(directory, file_name)
    key = " are selected."
    agent_dict = {}
    for file in files:
        with open(file, 'r') as f:
            lines = f.readlines()
            agent_list = []
            for idx, l in enumerate(lines):
                if key in l:
                    agent_list = json.loads(l.strip().replace(key, '').replace("'", '"'))
                    for agent in agent_list:
                        agent_dict[agent] = agent_dict.get(agent, 0) + 1
    print(agent_dict)
    sorted_new_data_set_v2 = dict(sorted(agent_dict.items(), key=lambda item: item[1], reverse=True))
    top_30_sorted_new_data_set_v2 = dict(list(sorted_new_data_set_v2.items())[:10])

    # Create the bar plot for the top 30 entries of the new data set
    plt.figure(figsize=(10, 12))
    plt.barh(list(top_30_sorted_new_data_set_v2.keys()), list(top_30_sorted_new_data_set_v2.values()), color='skyblue')
    plt.xlabel('Selected Times', fontsize=32)
    plt.ylabel('Agent Experts', fontsize=32)
    plt.xticks(fontsize=32)
    plt.yticks(fontsize=32)
    plt.xlim(0, 80)
    plt.gca().invert_yaxis()  # Highest values on top
    plt.savefig("math.svg", format="svg")
    plt.show()