source /home/cnets-vision/mengti_ws/dair_pll/cnets_env/bin/activate;
export PYTHONPATH=${PWD}:${PYTHONPATH}

dataset_sizes=(9)

for size in "${dataset_sizes[@]}"; do
    for run_idx in {10..15}; do
        cmd="xvfb-run python3 examples/bundlesdf_simple.py --structured --system=bundlesdf_cube --geometry=polygon --source=real --contactnets --regenerate --no-residual --loss-variation=1 --inertia-params=0 --dataset-size $size 'final_gt_mesh' 'final_gt_mesh-${run_idx}'"
        echo "Running: $cmd"
        eval $cmd
    done
done
