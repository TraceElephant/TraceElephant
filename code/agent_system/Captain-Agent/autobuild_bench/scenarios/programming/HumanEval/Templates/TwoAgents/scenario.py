import os
import json
import base64
import testbed_utils
import autogen

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

config1 = '__CONFIG_LIST_PATH__'
config2 = '__CONFIG_LIST_PATH2__'
##############################
logging_session_id = autogen.runtime_logging.start(config={"dbname": "logs.db"})
config_list = autogen.config_list_from_json(config1)

assistant = autogen.AssistantAgent(
    "assistant",
    is_termination_msg=lambda x: x.get("content", "").find("TERMINATE") >= 0,
    llm_config=testbed_utils.default_llm_config(config_list, timeout=180),
)

user_proxy = autogen.UserProxyAgent(
    "user_proxy",
    human_input_mode="NEVER",
    is_termination_msg=lambda x: x.get("content", "").find("TERMINATE") >= 0,
    code_execution_config={
        "work_dir": work_dir,
        "use_docker": False,
    },
    max_consecutive_auto_reply=10,
)

user_proxy.initiate_chat(
    assistant,
    message=f"""The following python code imports the `run_tests(candidate)` function from my_tests.py, and runs it on the function `__ENTRY_POINT__`. This will run a set of automated unit tests to verify the correct implementation of `__ENTRY_POINT__`. 
However, `__ENTRY_POINT__` is only partially implemented in the code below. 
Complete the implementation of `__ENTRY_POINT__` and output a new stand-alone code block that contains everything needed to run the tests, including: importing `my_tests`, calling `run_tests(__ENTRY_POINT__)`, as well as __ENTRY_POINT__'s complete definition, such that this code block can be run directly in Python.

```python
from my_tests import run_tests

{PROMPT}

# Run the unit tests. All unit tests are running online. DO NOT MODIFY THE FOLLOWING LINE.
run_tests(__ENTRY_POINT__)
```

The pass code should let `run_tests` function return "all test passed".""")
autogen.runtime_logging.stop()

##############################
testbed_utils.finalize(agents=[assistant, user_proxy])
