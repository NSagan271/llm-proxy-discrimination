#/bin/bash

# logfile with datatime
mkdir -p logs
LOGFILE_NAME=logs/run_llm_human_experiment_$(date +'%Y%m%d_%H%M%S').log

echo "Logging to $LOGFILE_NAME"

time python python_code/scripts/llm_human_experiment.py \
  --jsonl_input_file outputs/data/small.jsonl \
  --output_dir outputs/llm_human_experiment_test \
  --prompt_file prompts/staab_et_al_query_partial_attributes.txt \
  --model_names_file model_names.txt \
  --system_prompt_file prompts/staab_et_al_system.txt \
  --temperatures 0 \
  --max_threads 10 2> $LOGFILE_NAME
