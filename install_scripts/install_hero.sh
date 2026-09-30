#!/bin/bash
# Create a conda environment and install HERO.
#   bash install_scripts/install_hero.sh [env_name]
set -e
ENV_NAME=${1:-hero}

conda create -y -n "$ENV_NAME" python=3.10
eval "$(conda shell.bash hook)"
conda activate "$ENV_NAME"

pip install -e "$(dirname "$0")/../hero[dev]"
python "$(dirname "$0")/../check_environment.py"
