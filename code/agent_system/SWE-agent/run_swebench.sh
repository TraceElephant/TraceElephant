
sweagent run-batch \
    --config config/default.yaml \
    --agent.model.name gpt-4o \
    --num_workers 3 \
    --agent.model.per_instance_cost_limit 3.00 \
    --instances.type file \
    --instances.path sample_task/sampled_verified_test_100/my_instances.json \
    --use_structured_logging=True