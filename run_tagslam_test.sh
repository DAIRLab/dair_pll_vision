#!/bin/bash


### Experiments 6/3 to run PLL on BundleSDF trajectories with no vision geometry supervision.
source /mnt/data0/minghz/repos/bundlenets/dair_pll/pll_env/bin/activate
export PYTHONPATH=${PYTHONPATH}:/mnt/data0/minghz/repos/bundlenets/dair_pll
objs=("egg" "napkin" "cube" "bottle" "half" "milk")
nums=("1" "1-2" "1-3")
for obj in ${objs[@]}; do
    for num in ${nums[@]}; do
        if [ ! -d ./results/vision_${obj}/${obj}_${num}/tagslam/pll_id_09/geom_for_bsdf ]; then
            python examples/contactnets_vision.py --run-name=09 --vision-asset=${obj}_${num} --cycle-iteration=0 --w-bsdf=0 --skip-videos=all --clear-data
        else
	    echo "Already found results for ${obj}_${num}"
    	fi
        if [ ! -d ./results/vision_${obj}/${obj}_${num}/tagslam/pll_id_09/geom_for_bsdf ]; then
            python examples/restart_vision_run.py --run-name=09 --vision-asset=${obj}_${num} --cycle-iteration=0
        fi
    done
done

nums=("1-4" "1-5")
for obj in ${objs[@]}; do
    for num in ${nums[@]}; do
        if [ ! -d ./results/vision_${obj}/${obj}_${num}/tagslam/pll_id_09/geom_for_bsdf ]; then
            python examples/contactnets_vision.py --run-name=09 --vision-asset=${obj}_${num} --cycle-iteration=0 --w-bsdf=0 --skip-videos=all --clear-data
        else
	    echo "Already found results for ${obj}_${num}"
    	fi
        if [ ! -d ./results/vision_${obj}/${obj}_${num}/tagslam/pll_id_09/geom_for_bsdf ]; then
            python examples/restart_vision_run.py --run-name=09 --vision-asset=${obj}_${num} --cycle-iteration=0
        fi
    done
done


