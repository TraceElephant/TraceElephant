# README

## Data Prepare

You should download [GAIA](https://huggingface.co/datasets/gaia-benchmark/GAIA) and [AssistantBench](https://huggingface.co/datasets/AssistantBench/AssistantBench), and put dataset in this dir.

You can also get GAIA dataset here:

```shell
cd data
wget https://huggingface.co/datasets/miromind-ai/MiroFlow-Benchmarks/resolve/main/gaia-val.zip
unzip gaia-val.zip
# Unzip passcode: pf4*
```

## Code Change

If you use the default Bing search engine, it may trigger frequent bot verification; switching to DuckDuckGo can resolve this issue.

If you want to change the default search engine to duckduckgo, you can change code in `.venv/lib/python3.x/site-packages/autogen_ext/agents/web_surfer/_multimodal_web_surfer.py` like below:

```python
SEARCH_ENGINE = os.getenv("WEB_SURFER_SEARCH_ENGINE", "duckduckgo").lower()
SEARCH_ENGINES: dict[str, str] = {
    "duckduckgo": "https://duckduckgo.com/?q={query}",
    "bing": "https://www.bing.com/search?q={query}&FORM=QBLH",
}
```

## How to run

```bash
bash run_assistant_bench.sh

bash run_gaia.sh
```