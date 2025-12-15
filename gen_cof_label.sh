export CUDA_VISIBLE_DEVICES=3
python -m lcb_runner.runner.main \
    --model Qwen/Qwen3-Coder-30B-A3B-Instruct \
    --scenario codegeneration \
    --evaluate \
    --release_version release_v6 \
    --tensor_parallel_size 1 \
    --end_date 2024-08-01 \
    --n 16 \
    --temperature 0.3 \
    --cof v1 \
    --generate_funprm_training_data output/cof.json #Change to your CoF generated in previous step
