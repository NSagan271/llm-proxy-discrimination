#!/bin/bash

python python_code/scripts/build_synthpai.py \
  --data-dir data/synthpai \
  --output-dir outputs/data/synthpai \
  --all-samples \
  --max-texts-per-person 3 \
  --seed 42 \
  --test-prompt