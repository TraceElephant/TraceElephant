import json
from pathlib import Path


def find_files(directory, file_name):
    path = Path(directory)
    return list(path.rglob(file_name))

def remove_duplicate(agent_list):
    agent_list_temp = set(json.dumps(d, sort_keys=True) for d in agent_list)
    res_list = []
    for agent in agent_list_temp:
        res_list.append(json.loads(agent))
    
    return res_list

if __name__ == "__main__":
    directory = './'
    file_name_1 = 'build_history_*.json'
    file_name_2 = 'autobuild.json'
    file_meta_agent = find_files(directory, file_name_1)
    files_autobuild = find_files(directory, file_name_2)

    agent_list = []
    for file in files_autobuild:
        with open(file, 'r') as f:
            agent_list += json.load(f)['agent_configs']
    
    for file in file_meta_agent:
        with open(file, 'r') as f:
            agent_group = json.load(f)
            for group_name, configs in agent_group.items():
                agent_list += configs['agent_configs']
    
    agent_list = remove_duplicate(agent_list)
    
    json.dump(agent_list, open('agent_library.json', 'w'), indent=4)