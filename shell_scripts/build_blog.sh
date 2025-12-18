#!/bin/bash

python python_code/scripts/build_blog.py \
  --data-dir data/blogs \
  --output-dir outputs/data/blogs \
  --num-sample 600 \
  --seed 42 \
  --test-prompt