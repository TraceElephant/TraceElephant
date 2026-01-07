sweagent run \                                                   
--agent.model.name=gpt-4o \
--agent.model.per_instance_cost_limit=2.00 \
--env.repo.github_url=https://github.com/SWE-agent/test-repo \
--problem_statement.github_url=https://github.com/SWE-agent/test-repo/issues/1 \
--use_structured_logging=True