import os
import json
import autogen
import testbed_utils
from datetime import datetime

testbed_utils.init()

############################
logging_session_id = autogen.runtime_logging.start(config={"dbname": "logs.db"})

ANSWER = ""
with open("expected_answer.txt", "rt") as fh:
    ANSWER = fh.read()

GAIA_SYSTEM_MESSAGE = (
        "You are a helpful AI assistant, and today's date is "
        + datetime.now().date().isoformat()
        + """.
I will ask you a question. Answer this question using your coding and language skills.
You have access to the following environment variables: BING_API_key.
In the following cases, suggest python code (presented in a coding block beginning ```python) or shell script (presented in a coding block beginning ```sh) for the user to execute:
    1. When you need to collect info, use the code to output the info you need, for example, browse or search the web, download/read a file, print the content of a webpage or a file, check the operating system. After sufficient info is printed and the task is ready to be solved based on your language skill, you can solve the task by yourself.
    2. When you need to perform some task with code, use the code to perform the task and output the result. Finish the task smartly.
Answer the question step if you need to. If a plan is not provided, explain your plan first. Be clear which step uses code, and which step uses your language skill.
The user cannot provide any other feedback or perform any other action beyond executing the code appearing in the code block. The user can't modify your code, so do not suggest incomplete code which requires users to modify. Don't use a code block if it's not intended to be executed by the user. Don't include multiple code blocks in one response. Do not ask users to copy and paste code or results. Instead, use the 'print' function for the output when relevant. Check the execution result reported by the user.
If the result indicates there is an error, fix the error and output the code again. Suggest the full code instead of partial code or code changes. If the error can't be fixed or if the task is not solved even after the code is executed successfully, analyze the problem, revisit your assumption, collect additional info you need, and think of a different approach to try.
When you find an answer, report your thoughts, and finish your answer with the following template: FINAL ANSWER: [YOUR FINAL ANSWER].
YOUR FINAL ANSWER should be a number OR as few words as possible OR a comma separated list of numbers and/or strings.
If you are asked for a number, don't use comma to write your number neither use units such as $ or percent sign unless specified otherwise.
If you are asked for a string, don't use articles, neither abbreviations (e.g. for cities), and write the digits in plain text unless specified otherwise.
If you are asked for a comma separated list, apply the above rules depending of whether the element to be put in the list is a number or a string.
    """.strip()
)

filename = "__FILE_NAME__".strip()

question = """
__PROMPT__
""".strip()

if len(filename) > 0:
    question = f"Consider the file '{filename}', which can be read from the current working directory. If you need to read or write it, output python code in a code block (```python) to do so. {question}"

####################
config_list = autogen.config_list_from_json("OAI_CONFIG_LIST", filter_dict={"model": ["gpt-4-1106"]})
llm_config = testbed_utils.default_llm_config(config_list, timeout=180)

build_manager = autogen.OpenAIWrapper(config_list=config_list)
response_with_ans = build_manager.create(
    messages=[
        {
            "role": "user",
            "content": f"{GAIA_SYSTEM_MESSAGE}\n{question}",
        }
    ]
).choices[0].message.content

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

message_to_check = "Problem: " + question + f"\n\nReply: {response_with_ans}\n\nGround truth answer: " + ANSWER
checker_proxy.initiate_chat(answer_checker, message=message_to_check)
autogen.runtime_logging.stop()

####################
testbed_utils.finalize(agents=[answer_checker, checker_proxy])
