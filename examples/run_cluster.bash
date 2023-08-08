#!/bin/bash
#SBATCH --gpus=0
#SBATCH --mem-per-cpu=10G
#SBATCH --cpus-per-gpu=4
#SBATCH --qos=mp-med
#SBATCH --time=12:00:00
#SBATCH --job-name='pll'

export PYTHONPATH=${PWD}:${PYTHONPATH}

dataset_sizes=(4 8 16 32 64 128 256 512)

for size in "${dataset_sizes[@]}"; do
    for run_idx in {1..9}; do
        cmd="WANDB__SERVICE_WAIT=300 PYTHONUNBUFFERED=1 xvfb-run --server-args=\"-screen 0 800x600x24\" python3 examples/contactnets_simple.py --structured --system=cube --geometry=polygon --source=real --contactnets --regenerate --no-residual --dataset-size $size 'cube_$size' 'cube_${size}-${run_idx}'"
        echo "Running: $cmd"
        eval $cmd
    done
done
