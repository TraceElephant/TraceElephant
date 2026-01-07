import json
import autogen
import testbed_utils

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


question = """# Task instruction
- You need to consider a README carefully and write a python bash script to fulfill the user's need, taking care of the arguments in the script to match the user's instruction.
- You cannot run the python bash script or python code and testing them will have no feedbacks but only errors.
- Your final answer should be a single line python bash script.
- Do not suggest any code or scripts in ```...``` format. This will causes errors.

# README file
{readme}

# User instruction
{problem}

# Output format
>>> python YOUR ANSWER
"""

config1 = '__CONFIG_LIST_PATH__'
config2 = '__CONFIG_LIST_PATH2__'

####################
logging_session_id = autogen.runtime_logging.start(config={"dbname": "logs.db"})
config_list = autogen.config_list_from_json(config1)
llm_config = testbed_utils.default_llm_config(config_list, timeout=180)

assistant = autogen.AssistantAgent(
    "assistant",
    llm_config=llm_config,
    is_termination_msg=lambda x: x.get("content", "").find("TERMINATE") >= 0,
)

user_proxy = autogen.UserProxyAgent(
    "user_proxy",
    human_input_mode="NEVER",
    is_termination_msg=lambda x: x.get("content", "").find("TERMINATE") >= 0,
    code_execution_config={
        "work_dir": "coding",
        "use_docker": False,
    },
    max_consecutive_auto_reply=10,
)
user_proxy.initiate_chat(assistant, message=question.format(problem=PROBLEM, readme=README))


# --------- extract reply ---------
response_with_ans = ""
messages = assistant._oai_messages[user_proxy]
for j in range(len(messages) - 1, -1, -1):
    if (
        messages[j]["role"] == "assistant"
        and messages[j]["content"].strip() != "TERMINATE"
        and messages[j]["content"].strip() != "TERMINATE."
    ):
        response_with_ans = messages[j]["content"]
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
2. Check whether the answer in the reply matches the ground truth answer. When comparison is not obvious (for example, 3*\\sqrt(6) and 7.348), you may write code to check the answer and wait for the user to execute the code.
3. After everything is done, please choose a reply from the following options:
    - "The answer is correct."
    - "The answer is approximated but should be correct. Correct Answer: <ground truth answer> | Answer extracted: <answer extracted>."
    - "The answer is incorrect. Correct Answer: <ground truth answer> | Answer extracted: <answer extracted>."
    - "The reply doesn't contain an answer." """
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
message_to_check = "[Problem]: " + PROBLEM + f"\n[Reply]: {response_with_ans}\n\n[Ground truth arguments]: " + answer
checker_proxy.initiate_chat(answer_checker, message=message_to_check)
autogen.runtime_logging.stop()

####################
testbed_utils.finalize(agents=[assistant, user_proxy, answer_checker, checker_proxy])
