from typing import Dict, List, Optional, Union, Callable, Literal
import warnings

from autogen.agentchat.assistant_agent import ConversableAgent
from autogen.meta_scaffold_utils import MetaPromptingScaffolding
from autogen.agentchat import Agent
from autogen.oai.client import OpenAIWrapper

try:
    from termcolor import colored
except ImportError:

    def colored(x, *args, **kwargs):
        return x


class MetaPromptAgent(ConversableAgent):
    """
    (Experimental) An agent that uses the enhanced MetaPromptingScaffolding method to generate prompts for experts and manage the conversation with them.
    This method is detailed in the paper "Meta-Prompting: Enhancing Language Models with Task-Agnostic Scaffolding" by Suzgun et al. Pdf available at https://arxiv.org/abs/2401.12954
    The adaptations we introduced are:
    - Every expert is equipped with the ability to generate and execute Python code.
    - With the above enhancement, the Expert Python is no longer needed. Related prompt is removed.
    """

    meta_prompt_system_message = (
        '"You are an AI assistant that helps people find information. Please answer the following question.'
    )

    # This is used when the meta model does not provide a valid response
    error_message = 'If you have determined the final answer, please present it using the format below:\n\n>> FINAL ANSWER:\n"""\n[final answer]\n"""'

    final_answer_indicator = ">> FINAL ANSWER:"

    intermediate_feedback = "Based on the information given, what are the most logical next steps or conclusions? Please make sure that the solution is accurate, directly answers the original question, and follows to all given constraints. Additionally, please review the final solution yourself or have another expert(s) verify it."

    template_gen_expert_identity = """For each instruction, write a high-quality description about the most capable and suitable agent to answer the instruction. In second person perspective.

    [Instruction]: Make a list of 5 possible effects of deforestation.
    [Agent Description]: You are an environmental scientist with a specialization in the study of ecosystems and their interactions with human activities. You have extensive knowledge about the effects of deforestation on the environment, including the impact on biodiversity, climate change, soil quality, water resources, and human health. Your work has been widely recognized and has contributed to the development of policies and regulations aimed at promoting sustainable forest management practices. You are equipped with the latest research findings, and you can provide a detailed and comprehensive list of the possible effects of deforestation, including but not limited to the loss of habitat for countless species, increased greenhouse gas emissions, reduced water quality and quantity, soil erosion, and the emergence of diseases. Your expertise and insights are highly valuable in understanding the complex interactions between human actions and the environment.

    [Instruction]: Identify a descriptive phrase for an eclipse.
    [Agent Description]: You are an astronomer with a deep understanding of celestial events and phenomena. Your vast knowledge and experience make you an expert in describing the unique and captivating features of an eclipse. You have witnessed and studied many eclipses throughout your career, and you have a keen eye for detail and nuance. Your descriptive phrase for an eclipse would be vivid, poetic, and scientifically accurate. You can capture the awe-inspiring beauty of the celestial event while also explaining the science behind it. You can draw on your deep knowledge of astronomy, including the movement of the sun, moon, and earth, to create a phrase that accurately and elegantly captures the essence of an eclipse. Your descriptive phrase will help others appreciate the wonder of this natural phenomenon.

    [Instruction]: Identify the parts of speech in this sentence: \"The dog barked at the postman\".
    [Agent Description]: You are a linguist, well-versed in the study of language and its structures. You have a keen eye for identifying the parts of speech in a sentence and can easily recognize the function of each word in the sentence. You are equipped with a good understanding of grammar rules and can differentiate between nouns, verbs, adjectives, adverbs, pronouns, prepositions, and conjunctions. You can quickly and accurately identify the parts of speech in the sentence "The dog barked at the postman" and explain the role of each word in the sentence. Your expertise in language and grammar is highly valuable in analyzing and understanding the nuances of communication.
    """

    question_prefix: str = '''You are Meta-Expert, an extremely clever expert with the unique ability to collaborate with multiple experts (such as Expert Problem Solver, Expert Mathematician, Expert Essayist, etc.) to tackle any task and solve any complex problems. Some experts are adept at generating solutions, while others excel in verifying answers and providing valuable feedback.

    As Meta-Expert, your role is to oversee the communication between the experts, effectively using their skills to answer a given question while applying your own critical thinking and verification abilities.

    To communicate with a expert, type its name (e.g., "Expert Linguist" or "Expert Puzzle Solver"), followed by a colon ":", and then provide a detailed instruction enclosed within triple quotes. For example:

    Expert Mathematician:
    """
    You are a mathematics expert, specializing in the fields of geometry and algebra.
    Compute the Euclidean distance between the points (-2, 5) and (3, 7).
    """

    Ensure that your instructions are clear and unambiguous, and include all necessary information within the triple quotes. You can also assign personas to the experts (e.g., "You are a physicist specialized in...").

    Interact with only one expert at a time, and break complex problems into smaller, solvable tasks if needed. Each interaction is treated as an isolated event, so include all relevant details in every call.

    If you or an expert finds a mistake in another expert's solution, ask a new expert to review the details, compare both solutions, and give feedback. You can request an expert to redo their calculations or work, using input from other experts. Keep in mind that all experts, except yourself, have no memory! Therefore, always provide complete information in your instructions when contacting them. Since experts can sometimes make errors, seek multiple opinions or independently verify the solution if uncertain. Before providing a final answer, always consult an expert for confirmation. Ideally, obtain or verify the final solution with two independent experts. However, aim to present your final answer within 15 rounds or fewer.

    Refrain from repeating the very same questions to experts. Examine their responses carefully and seek clarification if required, keeping in mind they don't recall past interactions.

    Present the final answer as follows:
    >> FINAL ANSWER:
    """
    [final answer]
    """
    '''

    question_suffix: str = "\n\nLet's first come up with a list of experts you may want to consult for this problem and then immediately start solving it."

    # If expert prompting is enabled, the expert identity will be generated and appended, as well as the task description and input, to the prompt
    expert_prompting: bool = False

    meta_model_settings = {
        "temperature": 1,
        "top_p": 0.95,
        "max_tokens": 1024,
    }
    generator_settings = {
        "temperature": 1,
        "top_p": 0.95,
        "max_tokens": 1024,
    }
    code_execution_config = {
        "work_dir": "coding",
        "use_docker": False,
    }

    def __init__(
        self,
        name,
        is_termination_msg: Optional[Callable[[Dict], bool]] = None,
        max_consecutive_auto_reply: Optional[int] = None,
        code_execution_config: Union[Dict, Literal[False]] = False,
        llm_config: Optional[Union[Dict, Literal[False]]] = None,
        default_auto_reply: Optional[Union[str, Dict, None]] = "",
    ):
        if code_execution_config:
            self.code_execution_config = code_execution_config
        else:
            warnings.warn(
                "code_execution_config is set to False. However, code execution is required for MetaPromptAgent. Setting it to local execution."
            )

        super().__init__(
            name=name,
            is_termination_msg=is_termination_msg,
            max_consecutive_auto_reply=max_consecutive_auto_reply,
            human_input_mode="NEVER",
            code_execution_config=self.code_execution_config,
            llm_config=llm_config,
            default_auto_reply=default_auto_reply,
        )
        self.meta_model = MetaPromptingScaffolding(
            client=self.client,
            llm_config=llm_config,
            code_execution_config=self.code_execution_config,
            generator_settings=self.generator_settings,
            error_message=self.error_message,
            final_answer_indicator=self.final_answer_indicator,
            intermediate_feedback=self.intermediate_feedback,
        )
        self.register_reply([Agent, None], MetaPromptAgent.generate_meta_prompt_reply)

    def generate_meta_prompt_reply(
        self,
        messages: Optional[List[Dict]] = None,
        sender: Optional[Agent] = None,
        config: Optional[OpenAIWrapper] = None,
    ):
        """For now, we only consider the last message"""
        print(colored("Below is the inner conversation of the MetaPromptAgent", "yellow"), flush=True)

        message = messages[-1:].copy()
        input = message[0]["content"]
        message = [
            {
                "role": "system",
                "content": self.meta_prompt_system_message,
            }
        ]

        if self.expert_prompting:
            expert_messages = [
                {
                    "role": "user",
                    "content": f"{self.template_gen_expert_identity}\n\n[Instruction]:{input}\n[Agent Description]:",
                }
            ]

            # Generate the expert identity
            expert_identity = self.meta_model.generate(
                prompt_or_messages=expert_messages,
                **self.meta_model_settings,
            )

            # Create the text for the expert prompting using the expert identity, task description, and input
            template_expert_prompting = f"{expert_identity}\n\nNow given the above identity background, please answer the following question: {input}"  # TODO: task description

            # Append the expert prompting template to the prompt (message list)
            message.append(
                {
                    "role": "user",
                    "content": template_expert_prompting,
                }
            )
        else:
            # Append the task description and input to the prompt (message list)
            message.append(
                {
                    "role": "user",
                    "content": f"{self.question_prefix}Question: {input}{self.question_suffix}",
                }
            )

        # Get the full message log from the meta model
        message_log = self.meta_model.meta_model_generate(
            prompt_or_messages=message,
            max_tokens=self.meta_model_settings["max_tokens"],
            temperature=self.meta_model_settings["temperature"],
            top_p=self.meta_model_settings["top_p"],
            counter=0,
        )

        output = message_log[-1]["content"]

        return True, output
