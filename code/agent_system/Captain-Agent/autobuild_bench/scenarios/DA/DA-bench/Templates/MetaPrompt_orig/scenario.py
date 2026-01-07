import autogen
from autogen.agentchat.contrib.meta_prompting_orig import MetaPromptAgent
import re
import testbed_utils

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

PROMPT = """Let's solve a data analysis problem. 
Given an csv file path, you are required to answer a question following a constraint.

FILE PATH: ../data.csv

QUESTION: {question}

CONSTRAINT: {constraint}

After verification, reply with the final answer as the format of {formats}"""

ANSWER = ""
with open("expected_answer.txt", "rt") as fh:
    ANSWER = fh.read()

####################
logging_session_id = autogen.runtime_logging.start(config={"dbname": "logs.db"})
config_list = autogen.config_list_from_json("__CONFIG_LIST_PATH__")
llm_config = testbed_utils.default_llm_config(config_list, timeout=180)

user_proxy = autogen.UserProxyAgent(
    "user_proxy",
    human_input_mode="NEVER",
    is_termination_msg=lambda x: x.get("content", "").find("FINAL ANSWER") >= 0,
    code_execution_config={
        "work_dir": "coding",
        "use_docker": False,
    },
    max_consecutive_auto_reply=0,
    default_auto_reply="TERMINATE",
)

meta_prompt_agent = MetaPromptAgent(
    name="Metaprompt Agent",
    llm_config=llm_config,
    is_termination_msg=lambda x: x.get("content", "").find("TERMINATE") >= 0,
    code_execution_config={
        "work_dir": "coding",
        "use_docker": False,
    },
)

user_proxy.initiate_chat(meta_prompt_agent, message=PROMPT.format(question=QUESTION, constraint=CONSTRAINT, formats=FORMATS))

# --------- extract reply ---------
response_with_ans = ""
messages = meta_prompt_agent._oai_messages[user_proxy][-1]["content"]
pattern = "FINAL ANSWER:(.*)"
reply = re.findall(pattern, messages, re.DOTALL)
if len(reply) > 0:
    response_with_ans = reply[0].strip()

# --------- call LLM to check the answer ---------
check_sys_msg = """You are a helpful AI assistant. You will use your coding and language skills to verify the answer.
You are given:
    1. A problem.
    2. A reply with the answer to the problem.
    3. A ground truth answer.
Please do the following:
1. Extract the answer in the reply: "The answer is <answer extracted>".
2. Check whether the answer in the reply matches the ground truth answer. Only compare the values exist in ground truth answer.
3. Choose your answer from the following options:
    - "The answer is correct."
    - "The answer is approximated but should be correct. Correct Answer: <ground truth answer> | Answer extracted: <answer extracted>."
    - "The answer is incorrect. Correct Answer: <ground truth answer> | Answer extracted: <answer extracted>."
    - "The reply doesn't contain an answer." """

answer_checker = autogen.AssistantAgent(name="checker", llm_config=llm_config, system_message=check_sys_msg)
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

message_to_check = "Problem: " + QUESTION + f"\n\nReply: {response_with_ans}\n\nGround truth answer: " + ANSWER + "\n\nFormats:" + FORMATS
checker_proxy.initiate_chat(answer_checker, message=message_to_check)
autogen.runtime_logging.stop()
####################
testbed_utils.finalize(agents=[meta_prompt_agent, user_proxy, answer_checker, checker_proxy])
