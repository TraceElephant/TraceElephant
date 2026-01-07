import os
import json
import base64
import testbed_utils
import autogen
from autogen.agentchat.contrib.agent_builder import AgentBuilder

# NOTE:
# This scenario runs Human Eval in a slightly unconventional way:
# The agents have access to the unit tests, and can keep trying
# until they pass.

testbed_utils.init()
##############################

work_dir = "coding"

# Read the prompt
PROMPT = ""
with open("prompt.txt", "rt") as fh:
    PROMPT = fh.read()

AGENT_CONFIGS = ""
with open("agent_list.txt", "rt") as fh:
    AGENT_CONFIGS = fh.read()

####################
# Task parameters
max_agents = 10
config1 = '__CONFIG_LIST_PATH__'
config2 = '__CONFIG_LIST_PATH2__'
default_llm_config = {
    "temperature": 1,
    "top_p": 0.95,
    "max_tokens": 1024,
}

## build agents
logging_session_id = autogen.runtime_logging.start(config={"dbname": "logs.db"})
builder = AgentBuilder(config_file_or_env=config1,
                       builder_model_tags=["gpt-4", "0125", "1106", "claude3", "haiku"],
                       agent_model_tags=["gpt-4", "0125", "1106", "claude3", "haiku"],
                       max_agents=max_agents)
agent_list, agent_configs = builder.load(config_json=AGENT_CONFIGS)

## Run task
group_chat = autogen.GroupChat(agents=agent_list, messages=[], max_round=20, allow_repeat_speaker=agent_list[:-1] if agent_configs['coding'] is True else agent_list)
manager = autogen.GroupChatManager(
    groupchat=group_chat, code_execution_config={'use_docker': False}, llm_config={
        "config_list": autogen.config_list_from_json(config2, filter_dict={"tags": ["gpt-4", "0125", "1106", "claude3", "haiku"]}), 
        **default_llm_config
    }
)

agent_list[0].initiate_chat(manager, message=f"""
The following python code imports the `run_tests(candidate)` function from my_tests.py, and runs it on the function `__ENTRY_POINT__`. This will run a set of automated unit tests to verify the correct implementation of `__ENTRY_POINT__`. 
However, `__ENTRY_POINT__` is only partially implemented in the code below. 
Complete the implementation of `__ENTRY_POINT__` and output a new stand-alone code block that contains everything needed to run the tests, including: importing `my_tests`, calling `run_tests(__ENTRY_POINT__)`, as well as __ENTRY_POINT__'s complete definition, such that this code block can be run directly in Python.

```python
from my_tests import run_tests

{PROMPT}

# DO NOT MODIFY. 
# Run the unit tests
# It will return "all test passed" if the code pass all tests.
run_tests(__ENTRY_POINT__)
```
""")
autogen.runtime_logging.stop()
##############################
testbed_utils.finalize(agents=agent_list)
