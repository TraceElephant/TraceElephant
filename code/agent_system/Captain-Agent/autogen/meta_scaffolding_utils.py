# Import the necessary libraries
import os
import random
import re
from subprocess import Popen, PIPE, TimeoutExpired
import tempfile
import time
import retry
import warnings

from autogen import OpenAIWrapper
from autogen.code_utils import execute_code, extract_code

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
        # verifier_settings: Dict[str, Any],
        error_message: str,
        final_answer_indicator: str,
        expert_python_message: str,
        intermediate_feedback: str,
        fresh_eyes: bool = True,
        code_execution_config: Union[Dict, Literal[False]] = False,
        include_expert_name_in_instruction: bool = True,
        extract_output: bool = False,
        use_zero_shot_cot_in_expert_messages: bool = False,
    ) -> None:
        if code_execution_config is False:
            self.code_execution_config = {"work_dir": "coding", "use_docker": False, "timeout": 180}
            warnings.warn("Code execution config is not provided. Using default config.")
        else:
            self.code_execution_config = {
                "work_dir": code_execution_config.get("work_dir", "coding"),
                "use_docker": code_execution_config.get("use_docker", False),
                "timeout": code_execution_config.get("timeout", 180),
            }

        # Set the language model
        self.client = client
        self.llm_config = llm_config

        # Set the generator and verifier parameters + summarizer parameters (optional)
        self.generator_settings = generator_settings
        # self.verifier_settings = verifier_settings

        # Set the error message and final answer indicator
        self.error_message = error_message
        self.final_answer_indicator = final_answer_indicator

        # Set the fresh_eyes flag
        self.fresh_eyes = fresh_eyes

        # Other helper variables and constants for the model
        self.triple_quotes = '"""'

        # Set the include_expert_name_in_instruction flag
        self.include_expert_name_in_instruction = include_expert_name_in_instruction
        self.extract_output = extract_output
        self.expert_python_message = expert_python_message
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
                entire_message_log[-1]["content"] = f"ROUND {counter+1}:\n\n{entire_message_log[-1]['content']}"

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
                        f"ROUND {counter} Meta model output:\n",
                        "yellow",
                    )
                )
                print(meta_model_output)

                # Check if the meta_model_output contains a text of the form "Expert XYZ:\n" (where XYZ is an alphabanumeric string).

                # Step 2 (a): If we are not in the 0-shot CoT setting, check if the meta model output contains any text between triple quotes.
                # If it does, then generate an output from the corresponding model.
                pattern = r"Expert ((?:\w+ ?){1,5}):\n"
                if (self.fresh_eyes) and (
                    # f":\n{self.triple_quotes}" in meta_model_output
                    re.search(pattern, meta_model_output)
                ):
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

                            # By default, we use the generator Expert to generate an output from the instructions.
                            model_temp = self.generator_settings["temperature"]
                            model_top_p = self.generator_settings["top_p"]
                            model_max_tokens = self.generator_settings["max_tokens"]

                            current_model_message_list = [
                                {
                                    "role": "system",
                                    "content": 'You are an AI assistant that helps people find information. Please answer the following question. Once you have determined the final answer, please present it using the format below:\n\n>> FINAL ANSWER:\n"""\n[final answer]\n"""',
                                },
                                {
                                    "role": "user",
                                    "content": model_instruction,
                                },
                            ]

                            if model_name == "Expert Python":
                                current_model_message_list[-1][
                                    "content"
                                ] = f"{self.expert_python_message}.\n\n{current_model_message_list[-1]['content']}"

                            print(
                                colored(
                                    f"\nCalling {model_name}... Instruction:\n",
                                    "yellow",
                                )
                            )
                            print(model_instruction)

                            # Finally, read to call the corresponding model.
                            model_outputs = self.client.create(
                                messages=current_model_message_list,
                                max_tokens=model_max_tokens,
                                temperature=model_temp,
                                top_p=model_top_p,
                                **kwargs,
                            )
                            model_outputs = self.client.extract_text_or_completion_object(model_outputs)

                            for _, model_output in enumerate(model_outputs):
                                ## Special case for Expert Python
                                if model_name == "Expert Python":
                                    logs_all = ""
                                    code_blocks = extract_code(model_output)
                                    code_texts = []
                                    for i, code_block in enumerate(code_blocks):
                                        lang, code = code_block
                                        # By default, the language is python
                                        # TODO: Add code execution config
                                        exitcode, logs, _ = execute_code(
                                            code,
                                            timeout=self.code_execution_config["timeout"],
                                            work_dir=self.code_execution_config["work_dir"],
                                            use_docker=self.code_execution_config["use_docker"],
                                            lang="python",
                                        )
                                        logs_all += "\n" + logs
                                        code_texts.append(code)
                                    model_output += f"Here is the output of the code when executed:\n\n{logs_all}"

                                else:
                                    specicial_token = "* * *"
                                    if self.extract_output:
                                        # FIXME: Temporary fix
                                        if specicial_token in model_output:
                                            model_output = model_output.split(specicial_token)[1].strip()

                                        if len(model_output.split(" ")) > 128:
                                            model_output = "Solution too long. Please try again."
                                    else:
                                        model_output.replace(specicial_token, "")
                                        model_output.replace(
                                            "FINAL ANSWER:",
                                            f"{model_name}'s final answer:\n",
                                        )

                                intermediate_output += f"{model_name}'s output:\n{self.triple_quotes}\n{model_output}\n{self.triple_quotes}"
                                print(
                                    colored(
                                        f"\nResponse from {model_name}:\n",
                                        "yellow",
                                    )
                                )
                                print(model_output)

                            # Remove the last two newlines.
                            intermediate_output = intermediate_output.strip()

                    # Add the intermediate output to the full prompt or messages.
                    intermediate_output = f"{intermediate_output}\n\n{self.intermediate_feedback}"

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
            print(f"Houston, we have a problem in meta_model_generate: {e}")

            # If we have tried 7 times, then let's return the current prompt or messages.
            if trial_num == 7:
                return prompt_or_messages

            print("Waiting for 1-10 seconds...")
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
