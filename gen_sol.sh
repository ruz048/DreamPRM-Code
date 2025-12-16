python -m lcb_runner.runner.main \
    --model o4-mini__high \
    --scenario codegeneration \
    --evaluate \
    --release_version release_v6 \
    --start_date 2024-08-01 \
    --n 1 \
    --temperature 0.3 \
    --cof v1 \
    --max_tokens 4096 \
    --use_cache \
    --cache_batch_size 10
    