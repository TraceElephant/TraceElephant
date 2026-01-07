## Instruction
We use `autogenbench` to test all scenarios in our benchmark. For the detailed instruction of `autogenbench`, please refer to [autogenbench](https://microsoft.github.io/autogen/blog/2024/01/25/AutoGenBench/).
We also provided some brief instructions for `autogenbench` below.

## Installation
[Recommend] You can install pyautogen and autogenbench in editable way, which is easy to debug:
```bash
cd /path/to/autogen-autobuild-dev
pip install -e .
cd /path/to/autogen-bench-dev/samples/autogenbench
pip install -e .
```
You can also install official `pyautogen` and `autogenbench` by:
```bash
pip install pyautogen autogenbench
```
Because Meta-agent, Meta-prompting and the latest Autobuild has not been merged to main, so you need to modify the first line in `requirement.txt` to the path of your autogen-autobuild-dev.

## Usage
Use following command to run the benchmark for each scenario:
```bash
cd [SCENARIO FOLDER. For example, /path/to/scenarious/MATH]
python Scripts/init_tasks.py  // initialize the tasks
autogenbench run Tasks/[TASK YOU WANT TO RUN].jsonl --native  // run the task. native is use to run the scenario without docker. If you have a docker environment, you can remove it.
autogenbench tabulate Results/[TASK YOU WANT TO RUN]  // print the results in tabulate.
```

If you want to debug, set `-s 1` to use a single data for testing:
```bash
cd [SCENARIO FOLDER. For example, /path/to/scenarious/MATH]
autogenbench run Tasks/[TASK YOU WANT TO RUN].jsonl -s 1
```
If you want to debug a specific problem, you can run the `scenario.py` in `Results/[YOUR TASK]/[PROBLEM ID]/0/scenario.py` manually in debug mode.

Note that everytime the `autogenbench run TASK` will check the `Results` folder and only run the problems that are not in it. If you want to rerun the tasks, you need to delete the corresponding files in the `Results` folder.


## Contribute
To contribute to the benchmark, you need to prepare the following files:
- `Scripts/init_tasks.py`: This file is used to initialize the tasks for the benchmark, including dataset and prompt loading, and task json generation. You can define the substitution of the placeholder like `__PROMPT__` inside the Templates in the `init_tasks.py`.
- `Templates/[YOUR_TASK]/scenario.py`: The process of your method as a scenario.
- `Templates/[YOUR_TASK]/[EXTRA_FILES]`: The extra files needed for your method. For example, to record the results. 
- `MANIFEST.json`: including all files in this scenario.
- `README.md`: Should include the reference of the dataset and provide some brief instruction of how to run.
