# DA bench
## How to run
Download files in this [link](https://github.com/InfiAgent/InfiAgent/tree/main/examples/DA-Agent/data) to `autobuild_bench/scenarios/DA/data`.

Generate task jsonl files.
```bash
python Scripts/init_tasks.py
```
This will randomly select 20 tasks from the original task list and convert it to the desired format.

Run all the tasks.
```bash
echo 'Yes' | autogenbench run Tasks/two_agents.jsonl --native
```
