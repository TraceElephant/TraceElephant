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


# # Monkey patch: Fix Playwright Controller hard assertion issue
# def patch_playwright_controller():
#     """
#     Patch the click_id method of playwright_controller to avoid AssertionError
#     Replace hard assertions with exception handling to improve fault tolerance
#     """
#     try:
#         from autogen_ext.agents.web_surfer import playwright_controller
#         from playwright.async_api import Page

#         original_click_id = playwright_controller.PlaywrightController.click_id

#         async def patched_click_id(self, page, target_id):
#             """Patched click_id method, fault-tolerant handling for non-page navigation clicks"""
#             try:
#                 # Call original method
#                 result = await original_click_id(self, page, target_id)

#                 # If result is not a Page object, return current page instead of throwing exception
#                 if not isinstance(result, Page):
#                     print(f"[WARNING] Click on element {target_id} did not result in page navigation. Returning current page.")
#                     return page

#                 return result
#             except AssertionError:
#                 # Catch assertion error, return current page
#                 print(f"[WARNING] AssertionError caught when clicking element {target_id}. Link may be a download/popup/JS action. Returning current page.")
#                 return page

#         # Replace original method
#         playwright_controller.PlaywrightController.click_id = patched_click_id
#         print("[INFO] Playwright Controller patched successfully")

#     except ImportError as e:
#         print(f"[WARNING] Failed to patch Playwright Controller: {e}")




async def create_patchright_browser_context():
    """
    Create browser context using Patchright
    Patchright is a patched version of Playwright that removes automation features at the binary level
    More low-level and reliable than playwright-stealth
    """
    # Use Patchright instead of Playwright
    patchright = await async_patchright().start()

    # Launch browser (non-headless mode)
    # Patchright has automatically removed navigator.webdriver and other features, no additional configuration needed
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

    # Patchright doesn't need additional stealth plugins, it's already patched at the binary level
    return patchright, context


# GAIA system template
GAIA_SYSTEM_TEMPLATE = """Today's date is {today}.
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


class GAIABenchmarkRunner:
    def __init__(self, data_file: str, max_round: int = 20):
        self.data_file = data_file
        self.max_round = max_round
        # Allow overriding GAIA data root via env GAIA_DATA_DIR; fallback to data_file parent
        gaia_env = os.getenv("GAIA_DATA_DIR")
        self.gaia_dir = Path(gaia_env).expanduser().resolve() if gaia_env else Path(data_file).resolve().parent

        # Initialize OpenAI client
        self.client = OpenAIChatCompletionClient(
            model=os.getenv("M1_MODEL", "gpt-4o"),
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_API_BASE"),
            timeout=60.0  # Set API timeout to 60 seconds
        )

    def load_tasks(self) -> List[Dict[str, Any]]:
        """Load GAIA task data"""
        tasks = []
        with open(self.data_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    tasks.append(json.loads(line))
        return tasks

    def create_prompt(self, task: Dict[str, Any]) -> str:
        """Create task prompt"""
        today = datetime.now().strftime("%Y-%m-%d")
        question = task["task_question"]
        file_path = task.get("file_path")
        system_info = f"{platform.system()} {platform.release()} ({platform.machine()})"

        prompt = GAIA_SYSTEM_TEMPLATE.format(
            today=today,
            question=question.strip(),
            max_round=self.max_round,
            system_info=system_info
        )

        if file_path:
            # Build full file path
            full_path = self.gaia_dir / file_path
            prompt = (
                f"Consider the local file '{full_path}'. If helpful, read it with python code blocks. "
                f"Avoid asking the user to copy-paste content. {prompt}"
            )

        return prompt

    def approval_func(self, request: ApprovalRequest) -> ApprovalResponse:
        """Code execution approval function - auto-approve for unsupervised execution"""
        print(f"[AUTO-APPROVE] Code to execute:\n{request.code}")
        return ApprovalResponse(approved=True, reason="Auto-approved for benchmark execution")

    async def run_task(self, task_id: int, task: Dict[str, Any]) -> Dict[str, Any]:
        """Run a single task"""
        print(f"\n{'='*80}")
        print(f"Running Task {task_id}: {task['task_id']}")
        print(f"Question: {task['task_question'][:100]}...")
        print(f"{'='*80}\n")

        # Create prompt
        prompt = self.create_prompt(task)

        # Create browser context using Patchright
        # Patchright patches Chromium at the binary level, removing automation features more reliably than stealth plugins
        print("Creating Patchright browser context...")
        patchright, browser_context = await create_patchright_browser_context()

        # Configure WebSurfer to use Patchright context
        surfer = MultimodalWebSurfer(
            name="WebSurfer",
            model_client=self.client,
            playwright=patchright,  # Pass patchright instance
            context=browser_context,  # Pass browser context
            animate_actions=True,  # Add animation effects to simulate real users
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

        # Record end time
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        # Extract answer
        extracted_answer = self.extract_answer(messages)

        # Judge result
        is_correct = self.judge_answer(extracted_answer, task["ground_truth"])

        print(f"\n{'='*80}")
        print(f"Task {task_id} completed:")
        print(f"  Ground Truth: {task['ground_truth']}")
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
                # This may be an object containing complete conversation history returned at task end
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
                        # Try to find out the agent for next message
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
                    # Reply from other agents
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
            "question": task["task_question"],
            "ground_truth": task["ground_truth"],
            "question_ID": task.get("task_id", "unknown"),
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
        """Judge if the answer is correct"""
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

    parser = argparse.ArgumentParser(description="Run GAIA benchmark with MagenticOne")
    parser.add_argument("--data", default="gaia-val/standardized_data.jsonl", help="Path to GAIA data file")
    parser.add_argument("--start", type=int, default=0, help="Start task index")
    parser.add_argument("--end", type=int, default=None, help="End task index")
    parser.add_argument("--max-round", type=int, default=100, help="Maximum rounds per task")

    args = parser.parse_args()

    runner = GAIABenchmarkRunner(
        data_file=args.data,
        max_round=args.max_round
    )

    await runner.run_all_tasks(start_idx=args.start, end_idx=args.end)


if __name__ == "__main__":
    asyncio.run(main())
