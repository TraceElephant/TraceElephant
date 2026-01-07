import pandas as pd
from sentence_transformers import SentenceTransformer, util
from autogen.tool_utils import find_callables
from autogen.coding import LocalCommandLineCodeExecutor
from autogen import UserProxyAgent, AssistantAgent

class ToolBuilder:
    TOOL_USING_PROMPT_OLD = """

## Specific Python Functions
You have the access to the following useful python functions. They can be accessed from the python module called 'functions' by their function names.
If you want to import the python function called `foo` into your code, you can import it by writing `from functions import foo` in your code block.
Use them by following the function's instruction if you need.

{functions}
"""

    TOOL_USING_PROMPT = """

## Specific Python Functions
You have the access to the following useful python functions. They can be accessed from the python module called 'functions' by their function names.

**IMPORTANT**: Before using any of these functions, you MUST import them first.
For example, if you want to use a function called `foo`, you must write:
```python
from functions import foo
```
at the beginning of your code block.

Available functions:

{functions}

### Execution model
- Each code block you output is executed independently in a fresh command-line run. Imports, variables, and files created in one block are **not** automatically available in later blocks.
- Minimize the number of code blocks per response. Prefer solving the task in a single well-prepared block whenever possible.
- If you truly need multiple code blocks, every block must repeat the necessary `from functions import ...` statements at the top. Forgetting this causes `NameError: name 'function_name_xxx' is not defined`.

### Examples
**Correct (single block with import)**
```python
from functions import perform_web_search

print("Searching for harlequin shrimp paper...")
results = perform_web_search("Valencia-Mendez 2017 harlequin shrimp", count=5)
print(results)
```

**Incorrect (missing import ⇒ NameError)**
```python
# ❌ This will fail because perform_web_search was never imported.
results = perform_web_search("Valencia-Mendez 2017 harlequin shrimp", count=5)
print(results)
```

Remember: Always import the functions you need using `from functions import function_name` before calling them, in every code block you output.
Here is the English translation:

**Do not output both a code block and an answer in the same response:**
If you are already able to determine the answer, directly output the answer and the uppercase 'terminate'.
If you cannot determine the answer yet and need to run code, then when outputting a code block, do not output the answer or the uppercase 'terminate'.
"""

    def __init__(self, corpus_path, retriever, device="cpu"):

        self.df = pd.read_csv(corpus_path, sep='\t')
        document_list = self.df['document_content'].tolist()

        self.model = SentenceTransformer(retriever, device=device)
        self.embeddings = self.model.encode(document_list)
    
    def retrieve(self, query, top_k=3):
        # Encode the query using the Sentence Transformer model
        query_embedding = self.model.encode([query])

        hits = util.semantic_search(query_embedding, self.embeddings, top_k=top_k)
        
        results = []
        for hit in hits[0]:
            results.append(self.df.iloc[hit['corpus_id'], 1])
        return results
    
    def bind(self, agent: AssistantAgent, functions: str):
        """Binds the function to the agent so that agent is aware of it."""
        sys_message = agent.system_message
        sys_message += self.TOOL_USING_PROMPT.format(functions=functions)
        agent.update_system_message(sys_message)
        return
    
    def bind_user_proxy(self, agent: UserProxyAgent, tool_root: str):
        """
        Updates user proxy agent with a executor so that code executor can successfully execute function-related code.
        Returns an updated user proxy.
        """
        # Find all the functions in the tool root
        functions = find_callables(tool_root)

        code_execution_config = agent._code_execution_config
        executor = LocalCommandLineCodeExecutor(
            timeout=code_execution_config.get("timeout", 180),
            work_dir=code_execution_config.get("work_dir", "groupchat"),
            functions=functions
        )
        code_execution_config = {
            "executor": executor,
            "last_n_messages": code_execution_config.get("last_n_messages", 1)
        }
        updated_user_proxy = UserProxyAgent(
            name=agent.name,
            is_termination_msg=agent._is_termination_msg,
            code_execution_config=code_execution_config,
            human_input_mode="NEVER",
            default_auto_reply=agent._default_auto_reply
        )
        return updated_user_proxy
