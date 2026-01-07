import autogen
from autogen.agentchat.contrib.meta_prompting_agent_autogen import MetaPromptAgent
import re
import json
import testbed_utils

testbed_utils.init()

PROMPT = ""
with open("prompt.txt", "rt") as fh:
    PROMPT = fh.read()

README = ""
with open("readme.txt", "rt") as fh:
    README = fh.read()

ANSWER = ""
with open("expected_answer.txt", "rt") as fh:
    ANSWER = fh.read()
    ANSWER = json.loads(ANSWER)

question = f"""
Please solve the following machine learning development problem given by a user: 
{PROMPT}

You can refer to the following README:
{README}

You need to consider the README carefully and write a python bash script to fulfill the user's need, taking care of the arguments in the script to match the user's instruction.
In this task, you cannot run the python bash script or python code and testing them will have no feedbacks but only errors.
Your final answer should be a single line python bash script in the following format:

>>> python YOUR ANSWER

Do not suggest any code or scripts in ```...``` format. This will causes errors."""


####################
config1 = '__CONFIG_LIST_PATH__'
config2 = '__CONFIG_LIST_PATH2__'

####################
logging_session_id = autogen.runtime_logging.start(config={"dbname": "logs.db"})
    
config_list = autogen.config_list_from_json(config1)
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
    llm_config=llm_config.copy(),
    is_termination_msg=lambda x: x.get("content", "").find("TERMINATE") >= 0,
)

user_proxy.initiate_chat(meta_prompt_agent, message=PROMPT)


# --------- extract reply ---------
response_with_ans = ""
messages = meta_prompt_agent._oai_messages[user_proxy][-1]["content"]
pattern = "FINAL ANSWER:(.*)"
reply = re.findall(pattern, messages, re.DOTALL)
if len(reply) > 0:
    response_with_ans = reply[0].strip()


# --------- call LLM to check the answer 
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
config_list = autogen.config_list_from_json(config2)
llm_config = testbed_utils.default_llm_config(config_list, timeout=180)
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

answer = f"{' '.join([f'--{key} {value}' for key, value in ANSWER.items()])}"
message_to_check = "[Problem]: " + PROMPT + f"\n[Reply]: {response_with_ans}\n\n[Ground truth arguments]: " + answer
checker_proxy.initiate_chat(answer_checker, message=message_to_check)
autogen.runtime_logging.stop()
####################
testbed_utils.finalize(agents=[meta_prompt_agent, user_proxy, answer_checker, checker_proxy])
