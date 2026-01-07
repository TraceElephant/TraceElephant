# SCIBench Chemistry Benchmark

This scenario implements the [SCIBench Chemistry](https://github.com/mandyyyyii/scibench/tree/main) benchmark.

## Running the tasks

Prepare
```bash
cd autobuild_bench/scenarios/sci/Chemistry
git clone git@github.com:mandyyyyii/scibench.git ../scibench
python Scripts/init_tasks.py
```

```
autogenbench run Tasks/sci_chem_AutoBuild.jsonl
autogenbench tabulate Results/sci_chem_AutoBuild
```

By default, only a small subset (17 of 5000) MATH problems are exposed. Edit `Scripts/init_tasks.py` to expose more tasks.

**Note**: Answer has two versions: `answer_latex` and `answer_number`, I chose `answer_number` now.

## References
**SCIBENCH: Evaluating College-Level Scientific Problem-Solving Abilities of Large Language Models**<br/>
Xiaoxuan Wang, Ziniu Hu, Pan Lu, Yanqiao Zhu, Jieyu Zhang, Satyen Subramaniam, Arjun R. Loomba, Shichang Zhang, Yizhou Sun, Wei Wang<br/>
[https://arxiv.org/abs/2307.10635](https://arxiv.org/abs/2307.10635)
