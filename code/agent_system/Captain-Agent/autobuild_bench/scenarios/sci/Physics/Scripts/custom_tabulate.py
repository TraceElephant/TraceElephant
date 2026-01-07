import os
import sys
import json
from autogenbench.tabulate_cmd import default_tabulate


def scorer(instance_dir):
    reduced_problems = os.listdir("/linxindisk/linxin/llm/autogen-autobuild-dev/autobuild_bench/scenarios/sci/Physics/Results/sci_phy_AutoBuild")
    print(instance_dir)
    c = False
    for problem in reduced_problems:
        if str(problem) == instance_dir.split("/")[-2]:
            c = True
    if not c:
        return None
    checker_messages = os.path.join(instance_dir, "checker_messages.json")
    if os.path.isfile(checker_messages):
        with open(checker_messages, "rt") as fh:
            messages = json.loads(fh.read())["checker_proxy"]
            results = messages[-1]["content"].lower()
            if "the answer is correct" in results or "the answer is approximated but should be correct" in results:
                return True
            else:
                return False
    else:
        return None


def main(args):
    default_tabulate(args, scorer=scorer)


if __name__ == "__main__" and __package__ is None:
    main(sys.argv)
