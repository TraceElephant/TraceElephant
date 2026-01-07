import json
import autogen
import hashlib
from .agent_builder import AgentBuilder
from .tool_retriever import ToolBuilder
from typing import Callable, Dict, List, Literal, Optional, Union
from autogen.agentchat.conversable_agent import ConversableAgent
import os

from autogen.tool_utils import get_full_tool_description


class AutoBuildRetryableError(RuntimeError):
    """Raised when AutoBuild returns inconsistent or unusable output."""


def check_nested_mode_config(nested_mode_config: Dict):
    if "autobuild_init_config" in nested_mode_config.keys():
        assert (
            "autobuild_build_config" in nested_mode_config.keys()
        ), "autobuild_build_config is required when using autobuild as nested mode."
        assert (
            "group_chat_llm_config" in nested_mode_config.keys()
        ), "group_chat_llm_config is required when using autobuild as nested mode."
    elif "meta_prompting_llm_config" in nested_mode_config.keys():
        pass
    else:
        raise ValueError("nested_mode_config should contain either autobuild_init_config or meta_prompting_llm_config.")


class MetaUserProxyAgent(ConversableAgent):
    """(In preview) A proxy agent for the meta agent, that can execute code and provide feedback to the other agents."""

    CONVERSATION_REVIEW_PROMPT = """# Your task
Briefly summarize the conversation history derived from an experts' group chat by following the answer format.
If you found serious contradictions or issues in the conversation, point it out with a detailed reason and mark the "Need double-check" as "Yes."

# Conversation history:
{chat_history}

# Answer format
## Task
...

## Results
...

## Reason for the results
...

## Contradictions or issues in the conversation
...

### Need to double-check?
[Yes or No]

## Additional information (file path, code blocks, url, etc.)
...
"""

    AUTOBUILD_TASK_DESC = """You are given: (1) a task and advises from your manager with a specific plan and (2) a general task.
Collect information from the general task, follow the suggesstions from manager to solve the task.

# General Task
{general_task}

# Task and suggesstions from manager
{manager_task} """

    DEFAULT_AUTO_REPLY = "I'm a proxy and I can only execute your tool or end the conversation. If you think the problem is solved, please reply me only with 'TERMINATE'."

    # Default UserProxyAgent.description values, based on human_input_mode
    DEFAULT_USER_PROXY_AGENT_DESCRIPTIONS = {
        "ALWAYS": "An attentive HUMAN user who can answer questions about the task, and can perform tasks such as running Python code or inputting command line commands at a Linux terminal and reporting back the execution results.",
        "TERMINATE": "A user that can run Python code or input command line commands at a Linux terminal and report back the execution results.",
        "NEVER": "A computer terminal that can running Python scripts (provided to it quoted in ```python code blocks), or sh shell scripts (provided to it quoted in ```sh code blocks), or the conversation history and result of a group of agents",
    }

    def __init__(
        self,
        name: str,
        nested_mode_config: Dict,
        agent_config_save_path: str = None,
        is_termination_msg: Optional[Callable[[Dict], bool]] = None,
        max_consecutive_auto_reply: Optional[int] = None,
        human_input_mode: Optional[str] = "NEVER",
        function_map: Optional[Dict[str, Callable]] = None,
        code_execution_config: Optional[Union[Dict, Literal[False]]] = None,
        default_auto_reply: Optional[Union[str, Dict, None]] = DEFAULT_AUTO_REPLY,
        llm_config: Optional[Union[Dict, Literal[False]]] = False,
        system_message: Optional[Union[str, List]] = "",
        description: Optional[str] = None,
        default_group_name: Optional[str] = None,
    ):
        """
        Args:
            name (str): name of the agent.
            nested_mode_config (dict): the configuration for the nested chat mode.
                For autobuild, please refer to: autogen.agentchat.contrib.agent_builder[AgentBuilder]
                TODO: Add meta_prompting description
            is_termination_msg (function): a function that takes a message in the form of a dictionary
                and returns a boolean value indicating if this received message is a termination message.
                The dict can contain the following keys: "content", "role", "name", "function_call".
            max_consecutive_auto_reply (int): the maximum number of consecutive auto replies.
                default to None (no limit provided, class attribute MAX_CONSECUTIVE_AUTO_REPLY will be used as the limit in this case).
                The limit only plays a role when human_input_mode is not "ALWAYS".
            human_input_mode (str): whether to ask for human inputs every time a message is received.
                Possible values are "ALWAYS", "TERMINATE", "NEVER".
                (1) When "ALWAYS", the agent prompts for human input every time a message is received.
                    Under this mode, the conversation stops when the human input is "exit",
                    or when is_termination_msg is True and there is no human input.
                (2) When "TERMINATE", the agent only prompts for human input only when a termination message is received or
                    the number of auto reply reaches the max_consecutive_auto_reply.
                (3) When "NEVER", the agent will never prompt for human input. Under this mode, the conversation stops
                    when the number of auto reply reaches the max_consecutive_auto_reply or when is_termination_msg is True.
            function_map (dict[str, callable]): Mapping function names (passed to openai) to callable functions.
            code_execution_config (dict or False): config for the code execution.
                To disable code execution, set to False. Otherwise, set to a dictionary with the following keys:
                - work_dir (Optional, str): The working directory for the code execution.
                    If None, a default working directory will be used.
                    The default working directory is the "extensions" directory under
                    "path_to_autogen".
                - use_docker (Optional, list, str or bool): The docker image to use for code execution.
                    Default is True, which means the code will be executed in a docker container. A default list of images will be used.
                    If a list or a str of image name(s) is provided, the code will be executed in a docker container
                    with the first image successfully pulled.
                    If False, the code will be executed in the current environment.
                    We strongly recommend using docker for code execution.
                - timeout (Optional, int): The maximum execution time in seconds.
                - last_n_messages (Experimental, Optional, int): The number of messages to look back for code execution. Default to 1.
            default_auto_reply (str or dict or None): the default auto reply message when no code execution or llm based reply is generated.
            llm_config (dict or False): llm inference configuration.
                Please refer to [OpenAIWrapper.create](/docs/reference/oai/client#create)
                for available options.
                Default to false, which disables llm-based auto reply.
            system_message (str or List): system message for ChatCompletion inference.
                Only used when llm_config is not False. Use it to reprogram the agent.
            description (str): a short description of the agent. This description is used by other agents
                (e.g. the GroupChatManager) to decide when to call upon this agent. (Default: system_message)
            default_group_name (str): when provided, always reuse this group name so build artifacts can be shared across runs.
        """
        description = (
            description if description is not None else self.DEFAULT_USER_PROXY_AGENT_DESCRIPTIONS[human_input_mode]
        )
        super().__init__(
            name=name,
            system_message=system_message,
            is_termination_msg=is_termination_msg,
            max_consecutive_auto_reply=max_consecutive_auto_reply,
            human_input_mode=human_input_mode,
            function_map=function_map,
            code_execution_config=code_execution_config,
            llm_config=llm_config,
            default_auto_reply=default_auto_reply,
            description=description,
        )
        self.register_function(
            function_map={
                "seek_experts_help": lambda **args: self._run_autobuild(**args),
                "meta_prompting": lambda **args: self._run_meta_prompting(**args),
            }
        )
        check_nested_mode_config(nested_mode_config)
        self._agent_config_save_path = agent_config_save_path
        self._nested_mode_config = nested_mode_config.copy()
        self._code_execution_config = code_execution_config
        self.build_history = {}
        self.tool_history = {}
        self.build_times = 0
        self._default_group_name = default_group_name

    def _persist_build_history(self, building_task: str):
        if self._agent_config_save_path is not None:
            os.makedirs(self._agent_config_save_path, exist_ok=True)
            building_task_md5 = hashlib.md5(building_task.encode("utf-8")).hexdigest()
            with open(f"{self._agent_config_save_path}/build_history_{building_task_md5}.json", "w") as f:
                json.dump(self.build_history, f)

    def save_build_state(self, save_path: str, building_task: Optional[str] = None):
        """
        Persist the current build and tool history to disk so it can be reused without rebuilding.
        """
        save_dir = os.path.dirname(save_path) or "."
        os.makedirs(save_dir, exist_ok=True)
        state = {"build_history": self.build_history, "tool_history": self.tool_history}
        if building_task is not None:
            state["building_task"] = building_task
        with open(save_path, "w") as f:
            json.dump(state, f, indent=2)

    def load_build_state(self, load_path: str) -> Dict:
        """
        Load a previously saved build and tool history. This enables run-only mode.
        """
        with open(load_path, "r") as f:
            state = json.load(f)
        self.build_history = state.get("build_history", {})
        self.tool_history = state.get("tool_history", {})
        return state

    def _build_group(self, builder: AgentBuilder, group_name: str, building_task: str):
        retries = self._nested_mode_config.get("autobuild_build_config", {}).get("max_retries", 2)
        retries = max(0, int(retries))
        attempt = 0

        while True:
            try:
                agent_list, agent_configs = self._build_group_once(builder, group_name, building_task)
                self._persist_build_history(building_task)
                self.build_times += 1
                return agent_list, agent_configs
            except AutoBuildRetryableError as err:
                if attempt >= retries:
                    raise
                attempt += 1
                print(
                    f"[WARN] AutoBuild produced invalid output ({err}). Retrying attempt {attempt}/{retries}...",
                    flush=True,
                )

    def _build_group_once(self, builder: AgentBuilder, group_name: str, building_task: str):
        if group_name in self.build_history.keys():
            agent_list, agent_configs = builder.load(config_json=json.dumps(self.build_history[group_name]))
            if self._nested_mode_config.get("autobuild_tool_config", None) and agent_configs['coding'] is True:
                tool_root_dir = self._nested_mode_config["autobuild_tool_config"]["tool_root"]
                tool_builder = ToolBuilder(
                    corpus_path=self._nested_mode_config["autobuild_tool_config"]["tool_corpus"],
                    retriever=self._nested_mode_config["autobuild_tool_config"]["retriever"],
                )
                for idx, agent in enumerate(agent_list):
                    if idx == len(self.tool_history[group_name]):
                        break
                    tool_builder.bind(agent, "\n\n".join(self.tool_history[group_name][idx]))
                agent_list[-1] = tool_builder.bind_user_proxy(agent_list[-1], tool_root_dir)
            return agent_list, agent_configs

        build_config_snapshot = None
        if self._nested_mode_config["autobuild_build_config"].get('library_path_or_json', None):
            agent_list, agent_configs = builder.build_from_library(
                building_task, **self._nested_mode_config["autobuild_build_config"]
            )
            if not agent_list:
                raise AutoBuildRetryableError("AutoBuild returned no agents when building from the library.")
            build_config_snapshot = agent_configs.copy()

            if self._nested_mode_config.get("autobuild_tool_config", None) and agent_configs['coding'] is True:
                print("==> Retrieving tools...", flush=True)
                skills = [line for line in building_task.split("\n") if line.strip()] or [building_task]
                tool_builder = ToolBuilder(
                    corpus_path=self._nested_mode_config["autobuild_tool_config"]["tool_corpus"],
                    retriever=self._nested_mode_config["autobuild_tool_config"]["retriever"],
                )
                tool_root_dir = self._nested_mode_config["autobuild_tool_config"]["tool_root"]
                self._bind_tools_for_skills(agent_list, group_name, skills, tool_builder, tool_root_dir)
        else:
            agent_list, agent_configs = builder.build(
                building_task, **self._nested_mode_config["autobuild_build_config"]
            )
            if not agent_list:
                raise AutoBuildRetryableError("AutoBuild returned no agents when building from scratch.")
            build_config_snapshot = agent_configs.copy()

        if build_config_snapshot is not None:
            self.build_history[group_name] = build_config_snapshot

        return agent_list, agent_configs

    def _bind_tools_for_skills(self, agent_list, group_name, skills, tool_builder: ToolBuilder, tool_root_dir: str):
        if not agent_list:
            raise AutoBuildRetryableError("AutoBuild returned no agents to bind tools to.")

        new_tool_history = list(self.tool_history.get(group_name, []))
        pending_history = []
        for idx, skill in enumerate(skills):
            if idx >= len(agent_list):
                raise AutoBuildRetryableError(
                    f"AutoBuild produced {len(agent_list)} agents but needs at least {idx + 1} for the requested skills."
                )
            tools = tool_builder.retrieve(skill)
            docstrings = []
            for tool in tools:
                parts = tool.split()
                if len(parts) < 2:
                    raise AutoBuildRetryableError(
                        f"Invalid tool specification '{tool}' returned for skill '{skill}'."
                    )
                category, tool_name = parts[0], parts[1]
                tool_path = os.path.join(tool_root_dir, category, f"{tool_name}.py")
                try:
                    docstring = get_full_tool_description(tool_path)
                except FileNotFoundError as exc:
                    raise AutoBuildRetryableError(f"Tool file '{tool_path}' could not be found.") from exc
                except Exception as exc:
                    raise AutoBuildRetryableError(
                        f"Failed to read tool description for '{tool_path}': {exc}"
                    ) from exc
                docstrings.append(docstring)

            tool_builder.bind(agent_list[idx], "\n\n".join(docstrings))
            pending_history.append(docstrings)

        agent_list[-1] = tool_builder.bind_user_proxy(agent_list[-1], tool_root_dir)
        new_tool_history.extend(pending_history)
        self.tool_history[group_name] = new_tool_history

    def build_team(self, group_name: str, building_task: str) -> Dict:
        """
        Build only the expert team and bound tools without running the nested chat.
        Useful for a dedicated build phase that can be reused later.
        """
        builder = AgentBuilder(**self._nested_mode_config["autobuild_init_config"])
        group_name = self._default_group_name or group_name
        agent_list, agent_configs = self._build_group(builder, group_name, building_task)
        return {"group_name": group_name, "agent_count": len(agent_list), "agent_configs": agent_configs}
        self._default_group_name = default_group_name

    def _run_autobuild(self, group_name: str, execution_task: str, building_task: str = "") -> str:
        """
        Build a group of agents by AutoBuild to solve the task.
        This function requires the nested_mode_config to contain the autobuild_init_config,
            autobuild_llm_config, group_chat_llm_config.
        """
        group_name = self._default_group_name or group_name
        print("==> Running AutoBuild...", flush=True)
        print("\n==> Building task: ", building_task, flush=True)
        print("\n==> Execution task: ", execution_task, flush=True)

        builder = AgentBuilder(**self._nested_mode_config["autobuild_init_config"])
        agent_list, agent_configs = self._build_group(builder, group_name, building_task)

        # Store agent_list and group chat for later access to conversation history
        if not hasattr(self, '_agent_list_cache'):
            self._agent_list_cache = {}
            self._groupchat_cache = {}
        self._agent_list_cache[group_name] = agent_list

        # start nested chat

        def custom_speaker_selection(last_speaker, groupchat):
            """
            Prefer Computer_terminal when the last message contains executable code blocks.
            If last message contains a termination cue, end the conversation.
            Otherwise fall back to default "auto" selection.
            """
            import re
            if groupchat.messages:
                last_message = groupchat.messages[-1]
                content = last_message.get("content", "")
                if isinstance(content, str) and "TERMINATE" in content:
                    # stop the group chat
                    return None
                executable_code_pattern = r"```(python|py|sh|bash|shell)\n.*?```"
                if isinstance(content, str) and re.search(executable_code_pattern, content, re.DOTALL | re.IGNORECASE):
                    for agent in groupchat.agents:
                        if agent.name == "Computer_terminal":
                            print("🔧 Auto-selecting Computer_terminal to execute code", flush=True)
                            return agent
            return "auto"

        nested_group_chat = autogen.GroupChat(
            agents=agent_list,
            messages=[],
            allow_repeat_speaker=agent_list,
            speaker_selection_method=custom_speaker_selection,
            # speaker_selection_method="auto",
            **self._nested_mode_config["group_chat_config"],
        )
        manager = autogen.GroupChatManager(
            groupchat=nested_group_chat, llm_config=self._nested_mode_config["group_chat_llm_config"]
        )

        # Store the group chat object for later access
        self._groupchat_cache[group_name] = nested_group_chat
        key = list(self.chat_messages.keys())[0]
        general_task = self.chat_messages[key][0]['content']
        agent_list[0].initiate_chat(
            manager,
            message=self.AUTOBUILD_TASK_DESC.format(general_task=general_task, manager_task=execution_task)
        )
        chat_history = []
        key = list(agent_list[0].chat_messages.keys())[0]
        chat_messages = agent_list[0].chat_messages[key]
        for item in chat_messages:
            chat_history.append(item)

        # Review the group chat history.
        summary_model = builder.builder_model
        summarized_history = (
            summary_model.create(
                messages=[
                    {
                        "role": "user",
                        "content": self.CONVERSATION_REVIEW_PROMPT.format(chat_history=chat_history),
                    }
                ]
            )
            .choices[0]
            .message.content
        )

        return f"# Response from seek_agent_help: \n{summarized_history}"

    def _run_meta_prompting(self, expert_name: str, expert_identity: str, task: str) -> str:
        """
        Run Meta-prompting to solve the task.
        The method is adapted from "Meta-Prompting: Enhancing Language Models with Task-Agnostic Scaffolding".
        Paper available at https://arxiv.org/abs/2401.12954
        """
        print("Running meta prompting...")
        print("Querying expert: ", expert_name)

        expert = autogen.AssistantAgent(
            name=expert_name,
            human_input_mode="NEVER",
            llm_config=self._nested_mode_config["meta_prompting_llm_config"],
            system_message='You are an AI assistant that helps people find information. Please answer the following question. Once you have determined the final answer, please present it using the format below:\n\n>> FINAL ANSWER:\n"""\n[final answer]\n"""',
            max_consecutive_auto_reply=1,
        )
        user_proxy = autogen.UserProxyAgent(
            name="proxy",
            human_input_mode="NEVER",
            default_auto_reply="TERMINATE",
            code_execution_config=self._code_execution_config,
            max_consecutive_auto_reply=1,
        )
        task += "\nYou have access to python code interpreter. Suggest python code block starting with '```python' and the code will be automatically executed. The code will be executed exactly as they are, so do not suggest incomplete code which requires users to modify. You should always use print statement to get the value of a variable."
        user_proxy.initiate_chat(expert, message=expert_identity + "\n" + task, silent=True)

        expert_reply = user_proxy.chat_messages[expert][1]["content"]
        proxy_reply = user_proxy.chat_messages[expert][2]["content"]

        if proxy_reply != "TERMINATE":
            # Code is suggested by the expert
            code_result = proxy_reply[proxy_reply.find("Code output:") + len("Code output:") :].strip()
            expert_reply += f"\nThis is the output of the code blocks when executed:\n{code_result}"
        else:
            expert_reply.replace(
                "FINAL ANSWER:",
                f"{expert_name}'s final answer:\n",
            )

        return expert_reply
