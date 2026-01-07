# Captain Agent

These are the supplementary files for the "Adaptive In-conversation Team Building for Language Model Agents." They contain code for running the experiments in the paper.

The codebase is developed upon the AutoGen, where our implementations are located at `autogen/agentchat/contrib/meta_agent.py` and `autogen/agentchat/contrib/meta_user_proxy_agent.py`

## Run

Prereqs:
- GAIA/Assistant data at `dataset/`.
- An OpenAI config list in `OAI_CONFIG_LIST` (or pass `--config-list`).


You should download [GAIA](https://huggingface.co/datasets/gaia-benchmark/GAIA) and [AssistantBench](https://huggingface.co/datasets/AssistantBench/AssistantBench), and put data like this:

```shell
dataset
├── AssistantBench
│   └── assistant_bench_v1.0_dev.jsonl
└── gaia-val
    ├── 076c8171-9b3b-49b9-a477-244d2a532826.xlsx
    ├── 1f975693-876d-457b-a649-393859e79bf3.mp3
    ├── 2b3ef98c-cc05-450b-a719-711aee40ac65.mp3
    ├── 32102e3e-d12a-4209-9163-7b3a104efe5d.xlsx
    ├── 366e2f2b-8632-4ef2-81eb-bc3877489217.pdf
    ├── 389793a7-ca17-4e82-81cb-2b3a2391b4b9.txt
    ├── 3da89939-209c-4086-8520-7eb734e6b4ef.xlsx
    ├── 4d0aa727-86b1-406b-9b33-f870dd14a4a5.xlsx
    ├── 4d51c4bf-4b0e-4f3d-897b-3f6687a7d9f2.xlsx
    ├── 54612da3-fd56-4941-80f4-5eb82330de25.xlsx
    ├── 5b2a14e8-6e59-479c-80e3-4696e8980152.jpg
    ├── 5cfb274c-0207-4aa7-9575-6ac0bd95d9b2.xlsx
    ├── 6359a0b1-8f7b-499b-9336-840f9ab90688.png
    ├── 65afbc8a-89ca-4ad5-8d62-355bb401f61d.xlsx
    ├── 67e8878b-5cef-4375-804e-e6291fdbe78a.pdf
    ├── 7bd855d8-463d-4ed5-93ca-5fe35145f733.csv
    ├── 7bd855d8-463d-4ed5-93ca-5fe35145f733.xlsx
    ├── 7cc4acfa-63fd-4acc-a1a1-e8e529e0a97f.xlsx
    ├── 7dd30055-0198-452e-8c25-f73dbe27dcb8.pdb
    ├── 8d46b8d6-b38a-47ff-ac74-cda14cf2d19b.csv
    ├── 8f80e01c-1296-4371-9486-bb3d68651a60.png
    ├── 9318445f-fe6a-4e1b-acbf-c68228c9906a.png
    ├── 99c9cc74-fdc8-46c6-8f8d-3ce2d3bfeea3.mp3
    ├── 9b54f9d9-35ee-4a14-b62f-d130ea00317f.zip
    ├── a3fbeb63-0e8c-4a11-bff6-0e3b484c3e9c.pptx
    ├── b2c257e0-3ad7-4f05-b8e3-d9da973be36e.jpg
    ├── b7f857e4-d8aa-4387-af2a-0e844df5b9d8.png
    ├── bec74516-02fc-48dc-b202-55e78d0e17cf.jsonld
    ├── bfcd99e1-0690-4b53-a85c-0174a8629083.zip
    ├── c526d8d6-5987-4da9-b24c-83466fa172f3.xlsx
    ├── cca530fc-4052-43b2-b130-b30968d8aa44.png
    ├── cca70ce6-1952-45d2-acd4-80c903b0bc49.png
    ├── cffe0e32-c9a6-4c52-9877-78ceb4aaa9fb.docx
    ├── converted_audio.wav
    ├── d8152ad6-e4d5-4c12-8bb7-8d57dc10c6de.png
    ├── da52d699-e8d2-4dc5-9191-a2199e0b6a9b.xlsx
    ├── df6561b2-7ee5-4540-baab-5095f742716a.png
    ├── e9a2c537-8232-4c3f-85b0-b52de6bcba99.pdf
    ├── edd4d4f2-1a58-45c4-b038-67337af4e029.xlsx
    ├── extracted
    ├── f918266a-b3e0-4914-865d-4faa564f1aef.py
    ├── short_segment.mp3
    ├── short_segment.wav
    └── standardized_data.jsonl
```

You can also get GAIA dataset here:

```shell
cd data
wget https://huggingface.co/datasets/miromind-ai/MiroFlow-Benchmarks/resolve/main/gaia-val.zip
unzip gaia-val.zip
# Unzip passcode: pf4*
```


To build and run, run:

```bash
python scripts/run_gaia.py --mode build_and_run --task-index 0 --model gpt-4o
```

Just run using existing build config:

```bash
python scripts/run_gaia.py \
  --mode run_only \
  --task-index 0 \
  --build-state runs/gaia_task_0_20251124233348/build_state.json \
  --model gpt-4o
```

Useful flags:
- `--build-state`: where the build cache is stored/loaded (default `runs/captain_builds/gaia_captain.json`).
- `--building-task`: skill descriptions for the build phase.
- `--task-id`, `--task-level`, `-n`: select subsets of GAIA data.
- `--output`: JSONL file to collect answers (default `runs/gaia/results.jsonl`).
