config_list="OAI_CONFIG_LIST_0125"
config_list2="OAI_CONFIG_LIST_0125"
model="_0125"
# echo "*********************DA-Bench*********************"
# cd DA/DA-bench
# python Scripts/init_tasks.py --config-list $config_list --config-list2 $config_list2
# wait
# echo "*********************DA-Bench/MetaPrompt_autogen*********************"
# echo "Yes" | autogenbench run Tasks/da_MetaPrompt_autogen$model.jsonl --native -c $config_list
# wait
# echo "*********************DA-Bench/AutoBuild*********************"
# echo "Yes" | autogenbench run Tasks/da_AutoBuild$model.jsonl --native -c $config_list
# wait
# echo "*********************DA-Bench/TwoAgents*********************"
# echo "Yes" | autogenbench run Tasks/da_TwoAgents$model.jsonl --native -c $config_list
# wait
# echo "*********************DA-Bench/SingleLLM*********************"
# echo "Yes" | autogenbench run Tasks/da_SingleLLM$model.jsonl --native -c $config_list
# wait
# echo "*********************DA-Bench/MetaAgent*********************"
# echo "Yes" | autogenbench run "Tasks/da_MetaAgent$model.jsonl" --native -c $config_list
# wait

# echo "*********************MATH*********************"
# cd math/MATH
# python Scripts/init_tasks.py --config-list $config_list --config-list2 $config_list2
# wait
# echo "*********************MATH/MetaPrompt_autogen*********************"
# echo "Yes" | autogenbench run Tasks/math_MetaPrompt_autogen$model.jsonl --native -c $config_list
# wait
# echo "*********************MATH/TwoAgents*********************"
# echo "Yes" | autogenbench run Tasks/math_TwoAgents$model.jsonl --native -c $config_list
# wait
# echo "*********************MATH/AutoBuild*********************"
# echo "Yes" | autogenbench run Tasks/math_AutoBuild$model.jsonl --native -c $config_list
# wait
# echo "*********************MATH/SingleLLM*********************"
# echo "Yes" | autogenbench run Tasks/math_SingleLLM$model.jsonl --native -c $config_list
# wait
# echo "*********************MATH/MetaAgent*********************"
# echo "Yes" | autogenbench run "Tasks/math_MetaAgent$model.jsonl" --native -c $config_list
# wait

# echo "*********************HumanEval*********************"
# cd ../../programming/HumanEval
# python Scripts/init_tasks.py --config-list $config_list --config-list2 $config_list2
# wait
# echo "*********************HumanEval/MetaPrompt_autogen*********************"
# echo "Yes" | autogenbench run Tasks/r_human_eval_MetaPrompt_autogen$model.jsonl --native -c $config_list
# wait
# echo "*********************HumanEval/TwoAgents*********************"
# echo "Yes" | autogenbench run Tasks/r_human_eval_TwoAgents$model.jsonl --native -c $config_list
# wait
# echo "*********************HumanEval/AutoBuild*********************"
# echo "Yes" | autogenbench run Tasks/r_human_eval_AutoBuild$model.jsonl --native -c $config_list
# wait
# echo "*********************HumanEval/SingleLLM*********************"
# echo "Yes" | autogenbench run Tasks/r_human_eval_SingleLLM$model.jsonl --native -c $config_list
# wait
# echo "*********************HumanEval/MetaAgent*********************"
# echo "Yes" | autogenbench run "Tasks/r_human_eval_MetaAgent$model.jsonl" --native -c $config_list
# wait

echo "*********************SciBench/Chem*********************"
cd sci/Chemistry
python Scripts/init_tasks.py --config-list $config_list --config-list2 $config_list2
wait
echo "*********************SciBench/Chem/MetaPrompt_autogen*********************"
echo "Yes" | autogenbench run Tasks/sci_chem_MetaPrompt_autogen$model.jsonl --native -c $config_list
wait
echo "*********************SciBench/Chem/TwoAgents*********************"
echo "Yes" | autogenbench run Tasks/sci_chem_TwoAgents$model.jsonl --native -c $config_list
wait
echo "*********************SciBench/Chem/AutoBuild*********************"
echo "Yes" | autogenbench run Tasks/sci_chem_AutoBuild$model.jsonl --native -c $config_list
wait
echo "*********************SciBench/Chem/SingleLLM*********************"
echo "Yes" | autogenbench run Tasks/sci_chem_SingleLLM$model.jsonl --native -c $config_list
wait
echo "*********************SciBench/Chem/MetaAgent*********************"
echo "Yes" | autogenbench run "Tasks/sci_chem_MetaAgent$model.jsonl" --native -c $config_list
wait

echo "*********************SciBench/Phy*********************"
cd ../../sci/Physics
python Scripts/init_tasks.py --config-list $config_list --config-list2 $config_list2
wait
echo "*********************SciBench/Phy/MetaPrompt_autogen*********************"
echo "Yes" | autogenbench run Tasks/sci_phy_MetaPrompt_autogen$model.jsonl --native -c $config_list
wait
echo "*********************SciBench/Phy/TwoAgents*********************"
echo "Yes" | autogenbench run Tasks/sci_phy_TwoAgents$model.jsonl --native -c $config_list
wait
echo "*********************SciBench/Phy/AutoBuild*********************"
echo "Yes" | autogenbench run Tasks/sci_phy_AutoBuild$model.jsonl --native -c $config_list
wait
echo "*********************SciBench/Phy/SingleLLM*********************"
echo "Yes" | autogenbench run Tasks/sci_phy_SingleLLM$model.jsonl --native -c $config_list
wait
echo "*********************SciBench/Phy/MetaAgent*********************"
echo "Yes" | autogenbench run "Tasks/sci_phy_MetaAgent$model.jsonl" --native -c $config_list
wait
