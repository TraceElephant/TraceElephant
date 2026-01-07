from typing import Callable, Dict, Literal, Optional, Union
from autogen import AssistantAgent, ConversableAgent
from autogen.agentchat.contrib.tool_retriever import ToolBuilder
import warnings


class ToolAgent(AssistantAgent):
    tool_prompt = """# APIs
You have access to a number of apis. These apis should be called by a unifed python function.

## How to call apis
Function: call_api
Function Path: autogen/tool_utils.py
Description: Calls an API and returns the response. You should import it from the above path and call it.
Arguments:
    - api_name: string. The name of api to call. It should be exactly as provided by the documents.
    - api_input: string. The input to the api. It should be a string that can be converted to dictionary. For example, '{\n"id": "value"\n}'
Return: A string version of the response from called api.

## Available APIs
Below is the json schema of the available apis.
"""

    def __init__(
        self,
        name: str,
        tool_config: Optional[Union[Dict, Literal[False]]] = None,
        llm_config: Optional[Union[Dict, Literal[False]]] = None,
        is_termination_msg: Optional[Callable[[Dict], bool]] = None,
        max_consecutive_auto_reply: Optional[int] = None,
        human_input_mode: Optional[str] = "NEVER",
        description: Optional[str] = None,
        **kwargs
    ):
        super().__init__(
            name,
            llm_config=llm_config,
            is_termination_msg=is_termination_msg,
            max_consecutive_auto_reply=max_consecutive_auto_reply,
            human_input_mode=human_input_mode,
            description=description,
            **kwargs
        )
        if not tool_config:
            warnings.warn(
                "tool_config is not set. If you do not intend to use tool-enhanced assistant, please use AssistantAgent instead."
            )

        self.retriever = ToolBuilder(
            corpus_tsv_path=tool_config.get("corpus_path", "tools/corpus_dedup.tsv"),
            model_path=tool_config.get("model_path", "ToolIR/"),
            tool_path=tool_config.get("tool_path", "tools/toollib"),
        )
        self.top_k = tool_config.get("topk", 5)
        self.retrieved_tools = None
        self.register_reply(ConversableAgent, ToolAgent.generate_tool_augmented_reply)

    def generate_tool_augmented_reply(self, messages, **kwargs):
        message = messages[-1].copy()
        if self.retrieved_tools is None:
            print("Retrieving tools for conversation...")
            self.retrieved_tools = self.retriever.retrieve(message["content"], self.top_k)
            prompt = message["content"] + "\n" + self.tool_prompt + str(self.retrieved_tools)
            messages[-1]["content"] = prompt
            print(prompt)
        return self.generate_oai_reply(messages, **kwargs)
