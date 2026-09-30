#!/bin/bash
# Train the three HERO length specialists (128, 256 and 512 tokens) one after another.
# Extra arguments are forwarded to hero/train.py as Hydra overrides, e.g.
#   bash hero/scripts/train_specialists.sh seed=1 data.root=/path/to/data
set -e
cd "$(dirname "$0")/../.."

for LENGTH in 128 256 512; do
    python hero/train.py +exp=hero/specialist_${LENGTH} "$@"
done
