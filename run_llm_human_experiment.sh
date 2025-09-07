#/bin/bash

time python llm_human_experiment.py \
  --jsonl_input_file outputs/data/2025_09_06_14h31m51s/jsonl/synthpai_samples_trial_10.jsonl \
  --output_dir outputs/llm_human_experiment \
  --prompt_file prompts/staab_et_al_query_partial_attributes.txt \
  --system_prompt_file prompts/staab_et_al_system.txt \
  --model_names openai/o3 \
  --temperatures 0 \
  --top_p 0.9 \
  --max_threads 5