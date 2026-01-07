# Import the necessary libraries
import os
import random
import re
from subprocess import Popen, PIPE, TimeoutExpired
import tempfile
import time
import retry

from autogen import OpenAIWrapper, AssistantAgent, UserProxyAgent

# Import the typing library for type hints
from typing import Any, Dict, List, Optional, Tuple, Union, Literal

try:
    from termcolor import colored
except ImportError:

    def colored(x, *args, **kwargs):
        return x


# Define the MetaPromptingScaffolding class
class MetaPromptingScaffolding:
    def __init__(
        self,
        client: OpenAIWrapper,
        llm_config: List[Dict[str, Any]],
        generator_settings: Dict[str, Any],
        error_message: str,
        final_answer_indicator: str,
        intermediate_feedback: str,
        code_execution_config: Union[Dict, Literal[False]] = False,
        include_expert_name_in_instruction: bool = True,
        extract_output: bool = False,
        use_zero_shot_cot_in_expert_messages: bool = False,
    ) -> None:
        # Set the language model
        self.client = client
        self.llm_config = llm_config
        self.code_execution_config = code_execution_config

        # Set the generator and verifier parameters + summarizer parameters (optional)
        self.generator_settings = generator_settings

        # Set the error message and final answer indicator
        self.error_message = error_message
        self.final_answer_indicator = final_answer_indicator

        # Other helper variables and constants for the model
        self.triple_quotes = '"""'

        # Set the include_expert_name_in_instruction flag
        self.include_expert_name_in_instruction = include_expert_name_in_instruction
        self.extract_output = extract_output
        self.intermediate_feedback = intermediate_feedback
        self.use_zero_shot_cot_in_expert_messages = use_zero_shot_cot_in_expert_messages

    @retry.retry(tries=7, delay=5)
    def meta_model_generate(
        self,
        prompt_or_messages: Union[str, List[Dict[str, str]]],
        stop_tokens: Optional[List[str]] = None,
        max_tokens: int = 1024,
        temperature: float = 0.1,
        top_p: float = 0.95,
        counter: int = 0,
        last_answer: str = None,
        original_question: str = None,
        trial_num: int = 0,
        **kwargs: Any,
    ) -> Tuple[str, Any]:
        try:
            # This step is defined to ensure that the meta model returns a response in less than 16 rounds.
            # Note: Please feel free to change the number of rounds as you see fit.
            if counter == 16:
                return prompt_or_messages

            entire_message_log = prompt_or_messages.copy()

            while True:
                entire_message_log[-1]["content"] = f"\n==> ROUND {counter+1}:\n\n{entire_message_log[-1]['content']}"

                if counter == 14:
                    entire_message_log[-1]["content"] += "This is the last round; so, please present your final answer."

                # Step 1: Generate an output from the meta model
                meta_model_output = self.client.create(
                    messages=entire_message_log,
                    stop=stop_tokens,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    **kwargs,
                )
                meta_model_output = self.client.extract_text_or_completion_object(meta_model_output)[0]

                entire_message_log.append({"role": "assistant", "content": meta_model_output})
                print(
                    colored(
                        f"==> ROUND {counter} Meta model output:\n",
                        "yellow",
                    ),
                    flush=True
                )
                print(meta_model_output, flush=True)

                # Check if the meta_model_output contains a text of the form "Expert XYZ:\n" (where XYZ is an alphabanumeric string).

                # Step 2 (a): If we are not in the 0-shot CoT setting, check if the meta model output contains any text between triple quotes.
                # If it does, then generate an output from the corresponding model.
                pattern = r"Expert ((?:\w+ ?){1,5}):\n"
                if re.search(pattern, meta_model_output):
                    # There might be multiple instructions between the triple quotes; so, split the output by the triple quotes.
                    triple_quote_splits = meta_model_output.split(self.triple_quotes)
                    # Odd indices are the instructions, even indices contain the lines preceding the instructions (indicating which model to use).
                    len_triple_quote_splits = len(triple_quote_splits)

                    intermediate_output = ""
                    # Iterate over the instructions.
                    for i in range(1, len_triple_quote_splits, 2):
                        # Get the instructions for the corresponding model, as well as the line preceding the instructions (indicating which Expert to use).
                        line_preceding_instruction = triple_quote_splits[i - 1].strip()
                        model_name = line_preceding_instruction.split("\n")[-1].strip()
                        if "Expert " in model_name:
                            if model_name[-1] == ":":
                                model_name = model_name[:-1]

                            model_instruction = triple_quote_splits[i].strip()

                            # Add the expert name to the instruction.
                            if self.include_expert_name_in_instruction:
                                model_instruction = f"You are {model_name}.\n\n{model_instruction}"

                            # Add "Let's think step by step." to the instruction.
                            if self.use_zero_shot_cot_in_expert_messages:
                                model_instruction += "\n\nLet's think step by step."
                            model_instruction += "\n\nYou have access to python code interpreter. Suggest python code block starting with '```python' and the code will be automatically executed. You can use code to solve the task or for result verification. You should always use print statement to get the value of a variable."

                            # Define the expert agent as instructed by the meta model
                            expert = AssistantAgent(
                                name=model_name,
                                system_message='You are an AI assistant that helps people find information. Please answer the following question. Once you have determined the final answer, please present it using the format below:\n\n>> FINAL ANSWER:\n"""\n[final answer]\n"""',
                                llm_config=self.llm_config,
                                is_termination_msg=lambda x: x.get("content", "").find("TERMINATE") >= 0,
                                max_consecutive_auto_reply=1,
                            )
                            user_proxy = UserProxyAgent(
                                name=f"{model_name} proxy",
                                human_input_mode="NEVER",
                                code_execution_config=self.code_execution_config,
                                max_consecutive_auto_reply=1,
                                default_auto_reply="TERMINATE",
                            )
                            user_proxy.initiate_chat(expert, message=model_instruction, silent=True)

                            expert_reply = user_proxy.chat_messages[expert][1]["content"]
                            proxy_reply = user_proxy.chat_messages[expert][2]["content"]

                            if proxy_reply != "TERMINATE":
                                # Code is suggested by the expert
                                code_result = proxy_reply[
                                    proxy_reply.find("Code output:") + len("Code output:") :
                                ].strip()
                                expert_reply += (
                                    f"\n\nThis is the output of the code blocks when executed:\n\n{code_result}"
                                )
                            else:
                                expert_reply.replace(
                                    "FINAL ANSWER:",
                                    f"{model_name}'s final answer:\n",
                                )

                            print(
                                colored(
                                    f"\nResponse from {model_name}:\n",
                                    "yellow",
                                ),
                                flush=True
                            )
                            print(expert_reply, flush=True)

                            intermediate_output = f"{model_name}'s output:\n{self.triple_quotes}\n{expert_reply}\n{self.triple_quotes}".strip()

                    # Add the intermediate output to the full prompt or messages.
                    intermediate_output += f"\n\n{self.intermediate_feedback}"

                    # Add the intermediate output to the full prompt or messages.
                    entire_message_log.append(
                        {
                            "role": "user",
                            "content": intermediate_output,
                        }
                    )

                    # Prepare the prompt for the meta model
                    return self.meta_model_generate(
                        prompt_or_messages=entire_message_log,
                        stop_tokens=stop_tokens,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        counter=counter + 1,
                        last_answer=last_answer,
                        original_question=original_question,
                        **kwargs,
                    )
                # Step 2(b): Check if the meta_model_output contains the final answer indicator.
                elif self.final_answer_indicator in meta_model_output:
                    # The following code is commented out because we are not using the final answer indicator anymore.
                    # However, it is useful for debugging purposes.
                    # final_answer = meta_model_output.split(self.final_answer_indicator)[
                    #     -1
                    # ].strip()
                    # print(f"Final answer: {final_answer}")
                    return entire_message_log
                # Step 2(c): We need to continue the (meta-)conversation.
                else:
                    entire_message_log.append({"role": "user", "content": self.error_message})
                    return self.meta_model_generate(
                        prompt_or_messages=entire_message_log,
                        stop_tokens=stop_tokens,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        counter=counter + 1,
                        last_answer=last_answer,
                        original_question=original_question,
                        **kwargs,
                    )

        except Exception as e:
            print(f"Houston, we have a problem in meta_model_generate: {e}", flush=True)

            # If we have tried 7 times, then let's return the current prompt or messages.
            if trial_num == 7:
                return prompt_or_messages

            print("Waiting for 1-10 seconds...", flush=True)
            # Let's wait for 1-10 seconds before trying again.
            time.sleep(random.randint(1, 10))
            return self.meta_model_generate(
                prompt_or_messages=entire_message_log,
                stop_tokens=stop_tokens,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                counter=counter,
                last_answer=last_answer,
                trial_num=trial_num + 1,
                **kwargs,
            )

    @retry.retry(tries=7, delay=5)
    def generate(
        self,
        prompt_or_messages: Union[str, List[Dict[str, str]]],
        stop_tokens: Optional[List[str]] = None,
        max_tokens: int = 1024,
        temperature: float = 0.1,
        top_p: float = 0.95,
        **kwargs: Any,
    ) -> str:
        model_output = self.client.create(
            messages=prompt_or_messages,
            stop=stop_tokens,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            **kwargs,
        )
        model_output = self.client.extract_text_or_completion_object(model_output)[0]

        return model_output
