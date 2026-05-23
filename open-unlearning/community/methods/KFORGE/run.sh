#!/usr/bin/env bash
set -euo pipefail

python src/train.py \
  --config-name=unlearn \
  experiment=unlearn/tofu/kforge \
  task_name=tofu_kforge
