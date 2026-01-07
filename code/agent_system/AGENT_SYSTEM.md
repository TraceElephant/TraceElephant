# Agent System

Please note that if you want to collect agent execution traces during the process, you need to run the [LLM API Middleware](../llm_api_middleware/README.md).

## CaptainAgent

We use [Official implementation of CaptainAgent](https://github.com/LinxinS97/captain_agent_demo). We made a few minor modifications for usability. For example, the original search tool relied on the `BING_API`, which appears to have been deprecated, so we replaced it with an alternative search API. The core logic of the code remains the same as in the original implementation.


## Magentic-One

We use [Official implementation of Magentic-One](https://microsoft.github.io/autogen/stable//user-guide/agentchat-user-guide/magentic-one.html). We made a few minor modifications for usability. For example, when using the original Playwright Chromium, we frequently encountered bot-detection challenges, so we switched to Patchright to mitigate this issue. In addition, we chose to use [https://duckduckgo.com/](https://duckduckgo.com/) as the default search engine; compared to Bing, which often triggers CAPTCHA verification, DuckDuckGo did not present this problem during our usage.

## SWE-Agent

We use [Official implementation of SWE-Agent](https://github.com/SWE-agent/SWE-agent).