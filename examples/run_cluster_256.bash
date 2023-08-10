#!/bin/bash
#SBATCH --gpus=0
#SBATCH --mem-per-cpu=10G
#SBATCH --cpus-per-task=6
##SBATCH --cpus-per-gpu=4
#SBATCH --qos=mp-med
#SBATCH --time=12:00:00
#SBATCH --job-name='pll'

source /home/mengti/workspace/dair_pll/pll_env/bin/activate;
export PYTHONPATH=${PWD}:${PYTHONPATH}

dataset_sizes=(256)

for size in "${dataset_sizes[@]}"; do
    for run_idx in {4..9}; do
        cmd="WANDB__SERVICE_WAIT=300 PYTHONUNBUFFERED=1 xvfb-run --server-num=20 --server-args=\"-screen 0 800x600x24\" python3 examples/contactnets_simple.py --structured --system=cube --geometry=polygon --source=real --contactnets --regenerate --no-residual --loss-variation=1 --dataset-size $size 'gt_cube_$size' 'gt_cube_${size}-${run_idx}'"
        echo "Running: $cmd"
        eval $cmd
    done
done
