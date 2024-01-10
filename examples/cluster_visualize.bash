#!/bin/bash
#SBATCH --gpus=0
#SBATCH --mem-per-cpu=10G
#SBATCH --cpus-per-task=8
#SBATCH --qos=mp-med
#SBATCH --time=12:00:00
#SBATCH --job-name='pll'

source /home/mengti/workspace/dair_pll/venv/bin/activate;
export PYTHONPATH=${PWD}:${PYTHONPATH}


cmd="WANDB__SERVICE_WAIT=300 PYTHONUNBUFFERED=1 xvfb-run --server-args=\"-screen 0 800x600x24\" python3 test/test_contact_points.py"
echo "Running: $cmd"
eval $cmd