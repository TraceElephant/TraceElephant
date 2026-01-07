import asyncio
import json
import os
import platform
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any
from dotenv import load_dotenv
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_ext.teams.magentic_one import MagenticOne
from autogen_agentchat.ui import Console
from autogen_agentchat.agents import ApprovalRequest, ApprovalResponse, CodeExecutorAgent
from autogen_ext.agents.web_surfer import MultimodalWebSurfer
from autogen_ext.agents.file_surfer import FileSurfer
from autogen_ext.agents.magentic_one import MagenticOneCoderAgent
from autogen_agentchat.teams import MagenticOneGroupChat
from autogen_core.code_executor import CodeExecutor
from autogen_core.models import ChatCompletionClient
from typing import List
from autogen_agentchat.base import ChatAgent
from patchright.async_api import async_playwright as async_patchright

# Load environment variables
load_dotenv()


# Global variables for storing LLM call logs
llm_call_logs = []
current_steps_dir = None
_llm_client_patched = False


def patch_llm_client_for_logging():
    """
    Monkey patch OpenAI client's create method to log real LLM requests and responses
    Apply patch only once, using global variable current_steps_dir to determine save location
    """
    global _llm_client_patched

    if _llm_client_patched:
        return  # Already patched, don't apply again

    from autogen_ext.models.openai._openai_client import OpenAIChatCompletionClient

    original_create = OpenAIChatCompletionClient.create

    async def logged_create(self, messages, *args, **kwargs):
        """Wrapper method to log requests and responses"""
        import json
        from datetime import datetime

        # Call original method to get response
        response = await original_create(self, messages, *args, **kwargs)

        # If steps_dir is not set, don't log
        if current_steps_dir is None:
            return response

        # Build request record
        # Messages can be dicts or objects, need to handle both
        formatted_messages = []
        for msg in messages:
            if isinstance(msg, dict):
                role = msg.get("role", "unknown")
                content = msg.get("content", "")
            else:
                # If it's an object, try multiple possible attribute names
                # Check common role attribute names
                if hasattr(msg, "role"):
                    role = msg.role
                elif hasattr(msg, "source"):
                    role = msg.source
                elif hasattr(msg, "type"):
                    role = msg.type
                else:
                    role = "unknown"

                # Get content
                content = getattr(msg, "content", str(msg))

            # Handle multimodal content (may include non-serializable objects like Image)
            if isinstance(content, list):
                # Content is a list, may include text and images
                serializable_content = []
                image_counter = 0
                for item in content:
                    if isinstance(item, dict):
                        # If it's a dict, check if it contains an image
                        if item.get("type") == "image_url":
                            # Try to save image URL data
                            image_url = item.get("image_url", {})
                            if isinstance(image_url, dict):
                                url = image_url.get("url", "")
                                serializable_content.append({
                                    "type": "image_url",
                                    "image_url": {"url": url if len(url) < 200 else "<base64_data>"}
                                })
                            else:
                                serializable_content.append({
                                    "type": "image_url",
                                    "image_url": "<image_data>"
                                })
                        else:
                            serializable_content.append(item)
                    elif isinstance(item, str):
                        serializable_content.append({"type": "text", "text": item})
                    else:
                        # Try to save Image object
                        image_saved = False

                        # First check autogen_core._image.Image
                        try:
                            from autogen_core._image import Image as AutogenImage
                            if isinstance(item, AutogenImage):
                                image_counter += 1
                                step_num = len(llm_call_logs) + 1
                                image_filename = f"step_{step_num}_image_{image_counter}.png"
                                # Save to images subdirectory
                                image_path = current_steps_dir / "images" / image_filename

                                # autogen_core Image may have .image attribute storing PIL Image
                                if hasattr(item, 'image'):
                                    item.image.save(image_path)
                                elif hasattr(item, 'save'):
                                    item.save(image_path)
                                else:
                                    # Try to convert to PIL Image
                                    from PIL import Image as PILImage
                                    pil_img = PILImage.fromarray(item)
                                    pil_img.save(image_path)

                                serializable_content.append({
                                    "type": "image",
                                    # Use relative path in JSON
                                    "image_file": f"images/{image_filename}",
                                })
                                image_saved = True
                        except (ImportError, AttributeError, Exception):
                            pass

                        # If not autogen Image, check PIL Image
                        if not image_saved:
                            try:
                                from PIL import Image
                                if isinstance(item, Image.Image):
                                    image_counter += 1
                                    step_num = len(llm_call_logs) + 1
                                    image_filename = f"step_{step_num}_image_{image_counter}.png"
                                    # Save to images subdirectory
                                    image_path = current_steps_dir / "images" / image_filename
                                    item.save(image_path)
                                    serializable_content.append({
                                        "type": "image",
                                        # Use relative path in JSON
                                        "image_file": f"images/{image_filename}",
                                    })
                                    image_saved = True
                            except (ImportError, AttributeError, Exception):
                                pass

                        # If neither Image type or save failed, convert to string
                        if not image_saved:
                            serializable_content.append({"type": "text", "text": str(item)})
                content = serializable_content
            elif not isinstance(content, (str, int, float, bool, type(None))):
                # If not basic type, try to save as image or convert to string
                image_saved = False

                # First check autogen_core._image.Image
                try:
                    from autogen_core._image import Image as AutogenImage
                    if isinstance(content, AutogenImage):
                        step_num = len(llm_call_logs) + 1
                        image_filename = f"step_{step_num}_single_image.png"
                        # Save to images subdirectory
                        image_path = current_steps_dir / "images" / image_filename

                        # autogen_core Image may have .image attribute storing PIL Image
                        if hasattr(content, 'image'):
                            content.image.save(image_path)
                        elif hasattr(content, 'save'):
                            content.save(image_path)
                        else:
                            # Try to convert to PIL Image
                            from PIL import Image as PILImage
                            pil_img = PILImage.fromarray(content)
                            pil_img.save(image_path)

                        content = {
                            "type": "image",
                            # Use relative path in JSON
                            "image_file": f"images/{image_filename}",
                        }
                        image_saved = True
                except (ImportError, AttributeError, Exception):
                    pass

                # If not autogen Image, check PIL Image
                if not image_saved:
                    try:
                        from PIL import Image
                        if isinstance(content, Image.Image):
                            step_num = len(llm_call_logs) + 1
                            image_filename = f"step_{step_num}_single_image.png"
                            # Save to images subdirectory
                            image_path = current_steps_dir / "images" / image_filename
                            content.save(image_path)
                            content = {
                                "type": "image",
                                # Use relative path in JSON
                                "image_file": f"images/{image_filename}",
                            }
                            image_saved = True
                    except (ImportError, AttributeError, Exception):
                        pass

                # If neither Image type or save failed, convert to string
                if not image_saved:
                    content = str(content)

            formatted_messages.append({
                "role": role,
                "content": content
            })

        request_data = {
            "messages": formatted_messages,
            "model": kwargs.get("model", getattr(self, "_model", "gpt-4o")),
            "stream": kwargs.get("stream", False),
            "temperature": kwargs.get("temperature"),
            "top_p": kwargs.get("top_p"),
            "max_tokens": kwargs.get("max_tokens"),
        }

        # Remove None values
        request_data = {k: v for k, v in request_data.items() if v is not None}

        # Convert CreateResult to OpenAI ChatCompletion-like string for easier alignment with reference format
        def format_response_for_log(resp):
            try:
                from openai.types.chat import ChatCompletion, ChatCompletionMessage
                from openai.types.chat.chat_completion import Choice
                from openai.types.chat.chat_completion_message_tool_call import ChatCompletionMessageToolCall
                from openai.types.chat.chat_completion_message_function_tool_call import Function
                from openai.types.completion_usage import CompletionUsage
                from autogen_core.models import CreateResult

                if isinstance(resp, CreateResult):
                    usage = getattr(resp, "usage", None)
                    prompt_tokens = getattr(usage, "prompt_tokens", 0) if usage else 0
                    completion_tokens = getattr(usage, "completion_tokens", 0) if usage else 0

                    finish_reason = resp.finish_reason
                    if finish_reason == "function_calls":
                        finish_reason = "tool_calls"

                    tool_calls = None
                    message_content = None
                    if isinstance(resp.content, list):
                        tool_calls = []
                        for fc in resp.content:
                            tool_calls.append(
                                ChatCompletionMessageToolCall(
                                    id=getattr(fc, "id", ""),
                                    type="function",
                                    function=Function(
                                        arguments=getattr(fc, "arguments", ""),
                                        name=getattr(fc, "name", ""),
                                    ),
                                )
                            )
                    else:
                        message_content = "" if resp.content is None else str(resp.content)

                    message = ChatCompletionMessage(
                        role="assistant",
                        content=message_content,
                        tool_calls=tool_calls,
                        function_call=None,
                        refusal=None,
                        annotations=[],
                        audio=None,
                    )
                    choice = Choice(
                        finish_reason=finish_reason,
                        index=0,
                        logprobs=None,
                        message=message,
                    )
                    chat_completion = ChatCompletion(
                        id=f"chatcmpl-{uuid.uuid4().hex}",
                        choices=[choice],
                        created=int(datetime.now().timestamp()),
                        model=request_data.get("model", getattr(self, "_model", "unknown")),
                        object="chat.completion",
                        service_tier=None,
                        system_fingerprint=None,
                        usage=CompletionUsage(
                            prompt_tokens=prompt_tokens,
                            completion_tokens=completion_tokens,
                            total_tokens=prompt_tokens + completion_tokens,
                            completion_tokens_details=None,
                            prompt_tokens_details=None,
                        ),
                    )
                    return repr(chat_completion)
            except Exception as format_err:
                try:
                    print(f"[WARN] Failed to format response as ChatCompletion: {format_err}")
                except Exception:
                    pass

            # Fall back to default repr
            try:
                return repr(resp)
            except Exception:
                return str(resp)

        response_str = format_response_for_log(response)

        # Log to global list
        log_entry = {
            "request": request_data,
            "response": response_str,
            "timestamp": datetime.now().timestamp()
        }
        llm_call_logs.append(log_entry)

        # Immediately save to file
        step_num = len(llm_call_logs)
        step_file = current_steps_dir / f"step_{step_num}.json"
        with open(step_file, 'w', encoding='utf-8') as f:
            json.dump(log_entry, f, indent=2, ensure_ascii=False)

        return response

    # Replace original method
    OpenAIChatCompletionClient.create = logged_create
    _llm_client_patched = True
    print("[INFO] LLM Client patched for logging")


# Apply patch at program startup
patch_llm_client_for_logging()


async def create_patchright_browser_context():
    """
    Create browser context using Patchright
    Patchright is a patched version of Playwright that removes automation features at the binary level
    More low-level and reliable than playwright-stealth
    """
    # Use Patchright instead of Playwright
    patchright = await async_patchright().start()

    # Launch browser (non-headless mode)
    # Patchright automatically removes navigator.webdriver and other features, no additional configuration needed
    browser = await patchright.chromium.launch(
        headless=False,
        args=[
            '--disable-blink-features=AutomationControlled',
        ]
    )

    # Create browser context
    context = await browser.new_context(
        user_agent='Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36',
        viewport={'width': 1280, 'height': 720},
        locale='en-US',
    )

    # Patchright doesn't need additional stealth plugins, already patched at low level
    return patchright, context


# AssistantBench system template
ASSISTANT_BENCH_SYSTEM_TEMPLATE = """Today's date is {today}.
System: {system_info}. Do not use sudo if you need to run commands.

# Task
You need to solve the below question given by a user.

# Question
{question}

# Important Constraint
You MUST solve this problem within {max_round} rounds of conversation. Plan your approach efficiently and focus on the most direct path to the answer.

# Output format (MANDATORY)
Please respond in the following structure:
## ANSWER
[concise final answer]
## REASON
[brief reasoning or evidence used to reach the answer]

If you obtain and output the final answer, please output 'terminate' in **uppercase** format.
"""


class AssistantBenchRunner:
    def __init__(self, data_file: str, output_dir: str = "runs-assistant-bench", max_round: int = 20):
        self.data_file = data_file
        self.output_dir = Path(output_dir)
        self.max_round = max_round

        # Create output directory
        self.output_dir.mkdir(exist_ok=True)

        # Initialize OpenAI client
        self.client = OpenAIChatCompletionClient(
            model=os.getenv("M1_MODEL", "gpt-4o"),
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_API_BASE"),
            timeout=60.0  # Set API timeout to 60 seconds
        )

    def load_tasks(self) -> List[Dict[str, Any]]:
        """Load AssistantBench task data"""
        tasks = []
        with open(self.data_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    tasks.append(json.loads(line))
        return tasks

    def create_prompt(self, task: Dict[str, Any]) -> str:
        """Create task prompt"""
        today = datetime.now().strftime("%Y-%m-%d")
        question = task["task"]
        system_info = f"{platform.system()} {platform.release()} ({platform.machine()})"

        prompt = ASSISTANT_BENCH_SYSTEM_TEMPLATE.format(
            today=today,
            question=question.strip(),
            max_round=self.max_round,
            system_info=system_info
        )

        return prompt

    def approval_func(self, request: ApprovalRequest) -> ApprovalResponse:
        """Code execution approval function - auto-approve for unsupervised execution"""
        print(f"[AUTO-APPROVE] Code to execute:\n{request.code}")
        return ApprovalResponse(approved=True, reason="Auto-approved for benchmark execution")

    async def run_task(self, task_id: int, task: Dict[str, Any]) -> Dict[str, Any]:
        """Run a single task"""
        task_name = f"assistant_bench_task_{task_id}_{os.getenv('M1_MODEL', 'gpt-4o').replace('-', '_')}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        task_dir = self.output_dir / task_name
        task_dir.mkdir(exist_ok=True)

        # Create llm_steps directory
        steps_dir = task_dir / "llm_steps"
        steps_dir.mkdir(exist_ok=True)

        # Create images subdirectory
        images_dir = steps_dir / "images"
        images_dir.mkdir(exist_ok=True)

        # Set global steps_dir, let LLM patch log to this directory
        global llm_call_logs, current_steps_dir
        llm_call_logs = []  # Reset log list
        current_steps_dir = steps_dir

        print(f"\n{'='*80}")
        print(f"Running Task {task_id}: {task['id']}")
        print(f"Difficulty: {task.get('difficulty', 'N/A')}")
        print(f"Question: {task['task'][:100]}...")
        print(f"Output dir: {task_dir}")
        print(f"{'='*80}\n")

        # Create prompt
        prompt = self.create_prompt(task)

        # Create Patchright browser context
        # Patchright patches Chromium at low level, removes automation features, more reliable than stealth plugins
        print("Creating Patchright browser context...")
        patchright, browser_context = await create_patchright_browser_context()

        # Configure WebSurfer to use Patchright context
        surfer = MultimodalWebSurfer(
            name="WebSurfer",
            model_client=self.client,
            playwright=patchright,  # Pass patchright instance
            context=browser_context,  # Pass browser context
            animate_actions=True,  # Add animation effects to simulate real user
            start_page="https://duckduckgo.com/"
        )

        # Use complete MagenticOne (including all necessary agents)
        from autogen_ext.code_executors import create_default_code_executor
        code_executor = create_default_code_executor()

        fs = FileSurfer("FileSurfer", model_client=self.client)
        coder = MagenticOneCoderAgent("Coder", model_client=self.client)
        executor = CodeExecutorAgent("ComputerTerminal", code_executor=code_executor, approval_func=self.approval_func)

        m1 = MagenticOneGroupChat([fs, surfer, coder, executor], model_client=self.client)

        # Record start time
        start_time = datetime.now()

        # Run task and collect messages
        messages = []
        step_count = 0

        try:
            async for message in m1.run_stream(task=prompt):
                step_count += 1
                messages.append(message)

                # Print to console
                print(f"[Step {step_count}] {message}")

        except Exception as e:
            print(f"Error during task execution: {e}")
            messages.append({"error": str(e)})
        finally:
            # Clean up browser resources
            try:
                print("Cleaning up browser resources...")
                await browser_context.close()
                await patchright.stop()
            except Exception as cleanup_error:
                print(f"Error during cleanup: {cleanup_error}")

            # Reset global steps_dir (no need to declare global again, already declared above)
            current_steps_dir = None

        # Record end time
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        # Extract answer
        extracted_answer = self.extract_answer(messages)

        # Judge result
        is_correct = self.judge_answer(extracted_answer, task["answer"])

        # Format conversation history for summary.json
        history_data = self.format_history(messages, task)

        # Save summary.json (using same format as reference example)
        with open(task_dir / "summary.json", 'w', encoding='utf-8') as f:
            json.dump(history_data, f, indent=4, ensure_ascii=False)

        # Save judge.json
        judge_result = {
            "ground_truth": task["answer"],
            "extracted_answer": extracted_answer,
            "is_correct": is_correct,
            "reason": self.get_judge_reason(extracted_answer, task["answer"], is_correct)
        }

        with open(task_dir / "judge.json", 'w', encoding='utf-8') as f:
            json.dump(judge_result, f, indent=2, ensure_ascii=False)

        print(f"\n{'='*80}")
        print(f"Task {task_id} completed:")
        print(f"  Ground Truth: {task['answer']}")
        print(f"  Extracted Answer: {extracted_answer}")
        print(f"  Correct: {is_correct}")
        print(f"  Duration: {duration:.2f}s")
        print(f"  Total Steps: {step_count}")
        print(f"{'='*80}\n")

        # Return statistics
        return {
            "is_correct": is_correct,
            "duration_seconds": duration
        }

    def _extract_termination_reason(self, msg: Any, content: str, all_messages: List[Any]) -> str:
        """Extract termination reason from termination message"""
        # Try to extract termination reason from message object
        termination_reason = None

        # Check common termination reason attributes
        if hasattr(msg, 'stop_reason'):
            termination_reason = msg.stop_reason
        elif hasattr(msg, 'termination_condition'):
            termination_reason = msg.termination_condition
        elif hasattr(msg, 'reason'):
            termination_reason = msg.reason

        # If no termination reason found, try to infer from content
        if not termination_reason:
            if "Max turns" in content or "maximum number" in content:
                termination_reason = "Max turns reached"
            elif "Max time" in content or "timeout" in content.lower():
                termination_reason = "Max time reached"
            elif "No agent selected" in content or "stop_reason='no_termination'" in content:
                termination_reason = "No agent selected"
            elif "TERMINATE" in content or "terminate" in content:
                termination_reason = "Task completed"
            else:
                termination_reason = "Task ended"

        # Extract final answer from all messages (excluding current termination message)
        final_answer = self.extract_answer(all_messages[:-1] if all_messages else [])

        # Build termination message
        termination_message = f"{termination_reason}"

        # Add final answer
        if final_answer:
            termination_message += f"\nFINAL ANSWER: {final_answer}"
        else:
            termination_message += "\nFINAL ANSWER: Unable to determine"

        # Add completion marker
        termination_message += "\nCOMPLETE !#!#"

        return termination_message

    def format_history(self, messages: List[Any], task: Dict[str, Any]) -> Dict[str, Any]:
        """Format conversation history to standard format"""
        history = []
        previous_source = None

        for i, msg in enumerate(messages):
            # Extract message content and source
            if hasattr(msg, 'content'):
                content = msg.content if isinstance(msg.content, str) else str(msg.content)
            else:
                content = str(msg)

            # Detect and format termination message
            if content.startswith("messages=[") or content.startswith("[TextMessage(") or content.startswith("[MultiModalMessage("):
                # This may be an object containing full conversation history returned at task end
                # Extract termination reason and format to standard format
                termination_reason = self._extract_termination_reason(msg, content, messages)

                # Add termination condition message
                history.append({
                    "content": termination_reason,
                    "role": "Orchestrator (termination condition)"
                })
                continue

            # Determine role
            if hasattr(msg, 'source'):
                source = msg.source

                if source == 'user':
                    role = 'human'
                elif source == 'MagenticOneOrchestrator':
                    # Determine Orchestrator message type
                    if any(keyword in content for keyword in [
                        'GIVEN OR VERIFIED FACTS',
                        'Here is the plan',
                        'Updated Ledger',
                        'fact sheet',
                        'Next speaker'
                    ]):
                        role = 'Orchestrator (thought)'
                    else:
                        # This is an instruction to another agent
                        # Try to find the next message's agent
                        next_agent = None
                        if i + 1 < len(messages) and hasattr(messages[i + 1], 'source'):
                            next_agent = messages[i + 1].source
                            if next_agent != 'MagenticOneOrchestrator':
                                role = f'Orchestrator (-> {next_agent})'
                            else:
                                role = 'Orchestrator (thought)'
                        else:
                            role = 'Orchestrator'
                else:
                    # Other agent's reply
                    role = source
            else:
                role = 'unknown'

            history.append({
                "content": content,
                "role": role
            })

            previous_source = source if hasattr(msg, 'source') else None

        # Build complete history record
        history_data = {
            "history": history,
            "question": task["task"],
            "ground_truth": task["answer"],
            "question_ID": task.get("id", "unknown"),
            "is_corrected": False  # Default not corrected
        }

        return history_data

    def extract_answer(self, messages: List[Any]) -> str:
        """Extract answer from messages"""
        # Try to extract ## ANSWER section from last messages
        answer = ""
        for msg in reversed(messages):
            # Get message content - support different message types
            if hasattr(msg, 'content'):
                msg_str = msg.content if isinstance(msg.content, str) else str(msg.content)
            else:
                msg_str = str(msg)

            if "## ANSWER" in msg_str:
                # Extract ANSWER section
                lines = msg_str.split('\n')
                in_answer = False
                answer_lines = []
                for line in lines:
                    if "## ANSWER" in line:
                        in_answer = True
                        continue
                    if in_answer and line.startswith("##"):
                        break
                    if in_answer and line.strip():  # Only add non-empty lines
                        answer_lines.append(line.strip())
                answer = ' '.join(answer_lines).strip()
                if answer:
                    break

        return answer

    def judge_answer(self, extracted: str, ground_truth: str) -> bool:
        """Judge if answer is correct"""
        if not extracted:
            return False

        # Simple string matching (can be improved to more complex matching logic)
        extracted_lower = extracted.lower().strip()
        ground_truth_lower = ground_truth.lower().strip()

        return extracted_lower == ground_truth_lower or ground_truth_lower in extracted_lower

    def get_judge_reason(self, extracted: str, ground_truth: str, is_correct: bool) -> str:
        """Get judging reason"""
        if not extracted:
            return "Model answer is missing - cannot compare with ground truth"
        if is_correct:
            return "Extracted answer matches ground truth"
        return f"Extracted answer '{extracted}' does not match ground truth '{ground_truth}'"

    async def run_all_tasks(self, start_idx: int = 0, end_idx: int = None):
        """Run all tasks"""
        tasks = self.load_tasks()

        if end_idx is None:
            end_idx = len(tasks)

        print(f"Loaded {len(tasks)} tasks from {self.data_file}")
        print(f"Running tasks {start_idx} to {end_idx - 1}")

        results = []
        for i in range(start_idx, min(end_idx, len(tasks))):
            result = await self.run_task(i, tasks[i])
            results.append(result)

        # Calculate overall statistics
        total_tasks = len(results)
        correct_tasks = sum(1 for r in results if r["is_correct"])
        accuracy = correct_tasks / total_tasks if total_tasks > 0 else 0
        total_duration = sum(r["duration_seconds"] for r in results)

        print(f"\n{'='*80}")
        print(f"All tasks completed!")
        print(f"  Total: {total_tasks}")
        print(f"  Correct: {correct_tasks}")
        print(f"  Accuracy: {accuracy:.2%}")
        print(f"  Total Duration: {total_duration:.2f}s")
        print(f"{'='*80}\n")


async def main():
    """Main function"""
    import argparse

    parser = argparse.ArgumentParser(description="Run AssistantBench with MagenticOne")
    parser.add_argument("--data", default="AssistantBench/assistant_bench_v1.0_dev.jsonl", help="Path to AssistantBench data file")
    parser.add_argument("--output", default="runs-assistant-bench", help="Output directory for logs")
    parser.add_argument("--start", type=int, default=0, help="Start task index")
    parser.add_argument("--end", type=int, default=None, help="End task index")
    parser.add_argument("--max-round", type=int, default=100, help="Maximum rounds per task")

    args = parser.parse_args()

    runner = AssistantBenchRunner(
        data_file=args.data,
        output_dir=args.output,
        max_round=args.max_round
    )

    await runner.run_all_tasks(start_idx=args.start, end_idx=args.end)


if __name__ == "__main__":
    asyncio.run(main())
