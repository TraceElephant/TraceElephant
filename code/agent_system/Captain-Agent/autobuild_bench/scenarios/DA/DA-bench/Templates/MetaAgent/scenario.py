import autogen
import testbed_utils
from autogen.agentchat.contrib.meta_agent import MetaAgent
from autogen.agentchat.contrib.meta_user_proxy_agent import MetaUserProxyAgent

testbed_utils.init()

CONSTRAINT = ""
with open("constraint.txt", "rt") as fh:
    CONSTRAINT = fh.read()

FORMATS = ""
with open("format.txt", "rt") as fh:
    FORMATS = fh.read()

QUESTION = ""
with open("question.txt", "rt") as fh:
    QUESTION = fh.read()
    
config1 = '__CONFIG_LIST_PATH__'
config2 = '__CONFIG_LIST_PATH2__'

PROMPT = """Let's solve a data analysis problem. 
Given an csv file path, you are required to answer a question following a constraint. Do not plot any figure.

FILE PATH: ../data.csv

QUESTION: {question}

CONSTRAINT: {constraint}

After verification, reply with the final answer as the format of {formats}"""

ANSWER = ""
with open("expected_answer.txt", "rt") as fh:
    ANSWER = fh.read()

####################
# Task parameters
general_llm_config = {
    "max_tokens": 1500,
    "stream": False,
    "config_list": autogen.config_list_from_json(config2, filter_dict={"tags": ["gpt-4", "0125", "1106", "claude3", "haiku", "sonnet", "llama3"]}),
}
nested_mode_config = {
    "autobuild_init_config": {
        "config_file_or_env": config1,
        "builder_model_tags": ['gpt-4', '1106', '0125', 'claude3', 'haiku', 'sonnet', 'gemini-1.5', 'llama3', '8b', '70b', 'mixtral', '8x22b', '8x7b'],
        "agent_model_tags": ['gpt-4', '1106', '0125', 'claude3', 'haiku', 'sonnet', 'gemini-1.5', 'llama3', '8b', '70b', 'mixtral', '8x22b', '8x7b'],
    },
    "autobuild_build_config": {
        "default_llm_config": {
            "max_tokens": 1500,
            "cache_seed": None,
        },
        "coding": True,
        "library_path_or_json": "/linxindisk/linxin/llm/autogen-autobuild-dev/autobuild_bench/scenarios/agent_library.json",
    },
    "autobuild_tool_config": {
        "tool_corpus": "/linxindisk/linxin/llm/autogen-autobuild-dev/tools/tool_description.tsv",
        "tool_root": "/linxindisk/linxin/llm/autogen-autobuild-dev/tools",
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
    code_execution_config={
        "work_dir": "coding",
        "last_n_messages": 1
    },
    agent_config_save_path="__AGENT_SAVE_PATH__"
)

## Run task
meta_user_proxy.initiate_chat(
    meta_agent,
    message=PROMPT.format(question=QUESTION, constraint=CONSTRAINT, formats=FORMATS)
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
2. Check whether the answer in the reply matches the ground truth answer. Only compare the values exist in ground truth answer.
3. You **must** choose your answer from the following options:
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
        or "the answer is approximated but should be correct" in x.get("content", "").lower()
    ),
)

message_to_check = "Problem: [[\n" + QUESTION + f"]]\n\nReply: [[\n{response_with_ans}\n]]\n\nGround truth answer: [[\n" + ANSWER + "]]\n\nFormats: [[\n" + FORMATS + "]]"
checker_proxy.initiate_chat(answer_checker, message=message_to_check)
autogen.runtime_logging.stop()

####################
testbed_utils.finalize(agents=[meta_agent, meta_user_proxy, answer_checker, checker_proxy])
