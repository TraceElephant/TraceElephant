import autogen
from autogen.agentchat.contrib.meta_prompting_agent_autogen import MetaPromptAgent
import re
import testbed_utils

testbed_utils.init()

TABLE = ""
with open("table.txt", 'rt') as fh:
    TABLE = fh.read()

# PROMPT = "Please solve the following problem with tabular data:\n"
# with open("prompt.txt", "rt") as fh:
#     PROMPT += fh.read()
# PROMPT += "Table: "
# PROMPT += TABLE
# PROMPT += """\nPlease determine the type of question and answer accordingly.
# After verification, reply with the final answer in \\box{{}}.
# """
PROMPT = ""
with open("prompt.txt", "rt") as fh:
    PROMPT = fh.read()

ANSWER = ""
with open("expected_answer.txt", "rt") as fh:
    ANSWER = fh.read()

####################
config_list = autogen.config_list_from_json("OAI_CONFIG_LIST")
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
)

question = """Please answer the following problem with tabular data: 
{problem}
Table:
{table}
Please determine the type of question and answer accordingly.
After verification, reply with the final answer in \\box{{}}."""

user_proxy.initiate_chat(meta_prompt_agent, message=question.format(problem=PROMPT, table=TABLE))

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
2. Check whether the answer in the reply matches the ground truth answer. 
3. After everything is done, please choose a reply from the following options:
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
                                         or "the answer is approximated but should be correct" in x.get("content",
                                                                                                        "").lower()
                                 ),
)

message_to_check = "Problem: " + PROMPT + f"\n\nReply: {response_with_ans}\n\nGround truth answer: " + ANSWER
checker_proxy.initiate_chat(answer_checker, message=message_to_check)

####################
testbed_utils.finalize(agents=[meta_prompt_agent, user_proxy, answer_checker, checker_proxy])
