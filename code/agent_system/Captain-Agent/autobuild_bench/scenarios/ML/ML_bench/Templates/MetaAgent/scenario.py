import json
import autogen
import testbed_utils
from autogen.agentchat.contrib.meta_agent import MetaAgent
from autogen.agentchat.contrib.meta_user_proxy_agent import MetaUserProxyAgent

testbed_utils.init()

PROBLEM = ""
with open("prompt.txt", "rt") as fh:
    PROBLEM = fh.read()

README = ""
with open("readme.txt", "rt") as fh:
    README = fh.read()

ANSWER = ""
with open("expected_answer.txt", "rt") as fh:
    ANSWER = fh.read()
    ANSWER = json.loads(ANSWER)

config1 = '__CONFIG_LIST_PATH__'
config2 = '__CONFIG_LIST_PATH2__'
####################
## Task parameters
general_llm_config = {
    "temperature": 0,
    "config_list": autogen.config_list_from_json(
        config2, 
        filter_dict={
            "tags": ["gpt-4", "0125", "1106", "claude3", "haiku", "sonnet", "llama3"]
        }
    ),
}
nested_mode_config = {
    "autobuild_init_config": {
        "config_file_or_env": config1,
        "builder_model_tags": ['gpt-4', '1106', '0125', 'claude3', 'haiku', 'sonnet', 'gemini-1.5', 'llama3', '8b', '70b', 'mixtral', '8x22b', '8x7b'],
        "agent_model_tags": ['gpt-4', '1106', '0125', 'claude3', 'haiku', 'sonnet', 'gemini-1.5', 'llama3', '8b', '70b', 'mixtral', '8x22b', '8x7b'],
    },
    "autobuild_build_config": {
        "default_llm_config": {
            "temperature": 1,
            "top_p": 0.95,
            "max_tokens": 1500,
            "cache_seed": None,
        },
        "coding": True,
        "library_path_or_json": "__LIBRARY_PATH__",
    },
    "autobuild_tool_config": {
        "tool_corpus": "__TOOL_CORPUS__",
        "tool_root": "__TOOL_ROOT__",
        "retriever": "all-mpnet-base-v2",
    },
    "group_chat_config": {"max_round": 15},
    "group_chat_llm_config": general_llm_config.copy(),
}

## build agents
logging_session_id = autogen.runtime_logging.start(config={"dbname": "logs.db"})

meta_agent = MetaAgent(name="meta_agent", llm_config=general_llm_config.copy(), nested_mode="autobuild")
meta_user_proxy = MetaUserProxyAgent(
    name="meta_user_proxy",
    nested_mode_config=nested_mode_config,
    code_execution_config={},
    agent_config_save_path="__AGENT_SAVE_PATH__"
)

## Run task
question = """# README
{readme}

# User instruction
{problem}

# Task instruction
- You need to consider the README carefully and write a python bash script to fulfill the user's need, taking care of the arguments in the script to match the user's instruction.
- You cannot run the python bash script or python code and testing them will have no feedbacks but only errors.
- You don't need to verify the answer.
- Your final answer should be a single line python bash script.
- Do not suggest any code or scripts in ```...``` format. This will causes errors.

# Output format
>>> python YOUR ANSWER"""

meta_user_proxy.initiate_chat(
    meta_agent,
    message=question.format(problem=PROBLEM, readme=README)
)

## collect response
messages = []
key = list(meta_user_proxy.chat_messages.keys())[0]
chat_messages = meta_user_proxy.chat_messages[key]
for item in chat_messages:
    messages.append(item)
messages.reverse()

response_with_ans = "No answer."
for msg in messages:
    if (
        msg["content"] != "TERMINATE"
        and msg["content"] != "TERMINATE."
        and msg['role'] != 'assistant'
    ):
        response_with_ans = msg["content"]
        break

# ---------between "answer_checker" and "checker_proxy"---------
# define answer checker chat

check_sys_msg = """You are a helpful AI assistant. You will use your coding and language skills to compare the reply and answer.
You are given:
    1. A user instruction.
    2. A reply with the python bash script to the problem.
    3. Ground truth arguments for the script.
Please do the following:
1. Extract the python bash script in the reply: "The extracted python bash script is <answer extracted>".
2. Check whether the python bash script in the reply matches the ground truth python bash script. 
    - You need to carefully compare the arguments in the reply and answer. 
    - Additional arguments in the reply is allowed. But the arguments exist in the ground truth should be the same as in the reply.
3. After everything is done, please choose a reply from the following options:
    - "The answer is correct."
    - "The answer is approximated but should be correct. Correct Answer: <ground truth answer> | Answer extracted: <answer extracted>."
    - "The answer is incorrect. Correct Answer: <ground truth answer> | Answer extracted: <answer extracted>."
    - "The reply doesn't contain an answer." 
"""

answer_checker = autogen.AssistantAgent(
    name="checker",
    llm_config=general_llm_config.copy(),
    system_message=check_sys_msg
)
checker_proxy = autogen.UserProxyAgent(
    "checker_proxy",
    human_input_mode="NEVER",
    code_execution_config={
        "work_dir": "coding",
        "use_docker": False,
    },
    max_consecutive_auto_reply=5,
    default_auto_reply="TERMINATE",
    is_termination_msg=lambda x: x.get("content", "").lower()
    and (
        "the answer is correct" in x.get("content", "").lower()
        or "the answer is incorrect" in x.get("content", "").lower()
        or "the reply doesn't contain an answer" in x.get("content", "").lower()
        or "the answer is approximated but should be correct" in x.get("content", "").lower()
    ),
)

answer = f"{' '.join([f'--{key} {value}' for key, value in ANSWER.items()])}"
message_to_check = "[Problem]: " + PROBLEM + f"\n[Reply]: {response_with_ans}\n\n[Ground truth arguments]: " + answer
checker_proxy.initiate_chat(answer_checker, message=message_to_check)
autogen.runtime_logging.stop()

####################
testbed_utils.finalize(agents=[meta_agent, meta_user_proxy, answer_checker, checker_proxy])
