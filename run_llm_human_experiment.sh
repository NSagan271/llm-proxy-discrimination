#/bin/bash

time python llm_human_experiment.py \
  --jsonl_input_file outputs/data/2025_09_06_14h31m51s/jsonl/synthpai_samples_trial_1.jsonl \
  --output_dir outputs/llm_human_experiment \
  --prompt_file prompts/staab_et_al_query_partial_attributes.txt \
  --model_names deepseek/deepseek-chat-v3-0324 openai/o3 openai/gpt-4.1 anthropic/claude-sonnet-4 google/gemini-2.5-pro qwen/qwen3-max qwen/qwen3-30b-a3b meta-llama/llama-3.1-8b-instruct meta-llama/llama-3.1-70b-instruct mistralai/mixtral-8x22b-instruct \
  --system_prompt_file prompts/staab_et_al_system.txt \
  --temperatures 0 \
  --max_threads 10