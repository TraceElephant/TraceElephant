import autogen
import testbed_utils
from autogen.agentchat.contrib.meta_agent import MetaAgent
from autogen.agentchat.contrib.meta_user_proxy_agent import MetaUserProxyAgent

testbed_utils.init()

TABLE = ""
with open("table.txt", 'rt') as fh:
    TABLE = fh.read()

PROBLEM = ""
with open("prompt.txt", "rt") as fh:
    PROBLEM = fh.read()

ANSWER = ""
with open("expected_answer.txt", "rt") as fh:
    ANSWER = fh.read()

####################
# Task parameters
general_llm_config = {
    "temperature": 0,
    "config_list": autogen.config_list_from_json("OAI_CONFIG_LIST", filter_dict={"model": ["gpt-4-1106"]}),
}
nested_mode_config = {
    "autobuild_init_config": {
        "config_file_or_env": "OAI_CONFIG_LIST",
        "builder_model": "gpt-4-1106",
        "agent_model": "gpt-4-1106",
        "max_agents": 3,
    },
    "autobuild_build_config": {
        "default_llm_config": {
            "temperature": 1,
            "top_p": 0.95,
            "max_tokens": 1500,
        },
        "coding": True
    },
    "group_chat_config": {"max_round": 15},
    "group_chat_llm_config": general_llm_config.copy(),
}
## build agents
meta_agent = MetaAgent(name="meta_agent", llm_config=general_llm_config, nested_mode="autobuild")
meta_user_proxy = MetaUserProxyAgent(
    name="meta_user_proxy",
    nested_mode_config=nested_mode_config,
    code_execution_config={
        "use_docker": False,
    },
    agent_config_save_path="/Users/elpis/llm/autogen-autobuild-dev/autobuild_bench/scenarios/Tabular_Data/TabMWP/Saved_agents"
)

## Run task
question = """Please answer the following problem: 
{problem}

There is also some necessary information in the following table (you should provide the following table to the expert):
Table:
{table}

After verification, reply with the final answer in \\box{{}}."""

meta_user_proxy.initiate_chat(
    meta_agent,
    message=question.format(problem=PROBLEM, table=TABLE)
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

check_sys_msg = """You are a helpful AI assistant. You will use your coding and language skills to verify the answer.
You are given:
    1. A problem.
    2. A reply with the answer to the problem.
    3. A ground truth answer.
Please do the following:
1. Extract the answer in the reply: "The answer is <answer extracted>".
2. Check whether the answer in the reply matches the ground truth answer.
3. After everything is done, please choose a reply from the following options:
    - "The answer is correct."
    - "The answer is approximated but should be correct. Correct Answer: <ground truth answer> | Answer extracted: <answer extracted>."
    - "The answer is incorrect. Correct Answer: <ground truth answer> | Answer extracted: <answer extracted>."
    - "The reply doesn't contain an answer." """

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
                                         or "the answer is approximated but should be correct" in x.get("content",
                                                                                                        "").lower()
                                 ),
)

message_to_check = "[Problem]: " + PROBLEM + f"\n[Reply]: {response_with_ans}\n\n[Ground truth answer]: " + ANSWER
checker_proxy.initiate_chat(answer_checker, message=message_to_check)

####################
testbed_utils.finalize(agents=[meta_agent, meta_user_proxy, answer_checker, checker_proxy])
