#!/bin/bash
#SBATCH --gpus=0
#SBATCH --mem-per-cpu=10G
#SBATCH --cpus-per-gpu=4
#SBATCH --qos=mp-med
#SBATCH --partition=compute
#SBATCH --time=12:00:00
#SBATCH --job-name='pll'

# The list of dataset sizes you want to run the command for
dataset_sizes=(4 8 16 32 64 128 256 512)

# Loop through each dataset size
for size in "${dataset_sizes[@]}"; do
    # Loop for the run index
    for run_idx in {1..9}; do
        # Construct the command with the current dataset size and name
        cmd="WANDB__SERVICE_WAIT=300 PYTHONUNBUFFERED=1 xvfb-run --server-num="$SLURM_JOBID" --server-args="-screen 0 800x600x24" python examples/contactnets_simple.py --structured --system=cube --geometry=polygon --source=real --contactnets --regenerate --no-residual --dataset-size $size 'polygon_$size' 'polygon_${size}-${run_idx}'"
        echo "Running: $cmd"
        # Execute the command
        eval "$cmd"
    done
done
