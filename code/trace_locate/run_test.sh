cd "$(dirname "$0")"

./run_grid_parallel.sh \
    --api-key "sk-xxx" \
    --model "gpt-4o-mini" \
    --base-url "https://api.xxx.com/v1" \
    --data-dir "/path/to/data" \
    --output-dir "outputs" \
    --max-workers 20 \
    --methods "all_at_once"
