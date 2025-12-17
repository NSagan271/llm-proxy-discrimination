#/bin/bash

time python python_code/scripts/llm_human_experiment.py \
  --dataset_dir outputs/data/synthpai_small \
  --output_dir outputs/llm_human_experiment_test \
  --prompt_file prompts/staab_et_al_query_partial_attributes.txt \
  --model_names_file model_names_small.txt \
  --system_prompt_file prompts/staab_et_al_system.txt \
  --temperatures 0 \
  --max_threads 10
