#!/bin/bash

# 00: new data new mask

### iter 0 (tagslam trajectory)
# python examples/contactnets_vision.py  --run-name=00 --vision-asset=cube_1 --cycle-iteration=0 --clear-data #--bundlesdf-id=B
# python examples/contactnets_vision.py  --run-name=00 --vision-asset=cube_2 --cycle-iteration=0 --clear-data 
# python examples/contactnets_vision.py  --run-name=00 --vision-asset=cube_3 --cycle-iteration=0 --clear-data 
# python examples/contactnets_vision.py  --run-name=00_monitor_train_trajerror --vision-asset=cube_4 --cycle-iteration=0 #--clear-data 

# cube, bottle, half, milk, toblerone, prism, egg, napkin, box
# bakingbox, burger, cardboard, chocolate, cream, croc, crushedcan, duck, gallon, greencan, hotdog, icetray, mug, oatly, pinkcan, stapler, styrofoam, toothpaste

### Write a loop for below:
# bakingbox, burger, cardboard, chocolate, cream, croc, crushedcan, duck, gallon, greencan, hotdog, icetray, mug, oatly, pinkcan, stapler, styrofoam, toothpaste
# objs=("bakingbox" "burger" "cardboard" "chocolate" "cream" "croc" "crushedcan" "duck" "gallon" "greencan" "hotdog" "icetray" "mug" "oatly" "pinkcan" "stapler" "styrofoam" "toothpaste")

# cube, bottle, half, milk, toblerone, prism, egg, napkin, box
# rerun bottle_1-4
#objs=("bakingbox" "burger" "cardboard" "chocolate" "cream" "croc" "crushedcan" "duck" "gallon" "greencan" "hotdog" "icetray" "mug" "oatly" "pinkcan" "stapler" "styrofoam" "toothpaste" "prism" "egg" "napkin" "cube" "bottle" "half" "milk" "toblerone" )
#nums=("1-4" "1-5") # ("1" "1-2" "1-3" ) # "1-4" "1-5"
#for obj in ${objs[@]}; do
#    for num in ${nums[@]}; do
#        python examples/contactnets_vision.py  --run-name=02 --vision-asset=${obj}_${num} --cycle-iteration=2 --bundlesdf-id=02 --skip-videos all --clear-data #--bundlesdf-id=00
#        python examples/restart_vision_run.py  --run-name=02 --vision-asset=${obj}_${num} --cycle-iteration=2
#    done
#done
# python examples/contactnets_vision.py  --run-name=00 --vision-asset=cube_1 --cycle-iteration=0 --skip-videos all --clear-data

# ### change output without rerunning pll (need to reconfigure the file)
# python bundlesdf_interface.py


# ### Experiments 6/1 to run PLL on BundleSDF trajectories with no vision geometry supervision.
# source /mnt/data0/minghz/repos/bundlenets/dair_pll/pll_env/bin/activate
# export PYTHONPATH=${PYTHONPATH}:/mnt/data0/minghz/repos/bundlenets/dair_pll
# objs=("bakingbox" "cardboard" "crushedcan" "gallon" "greencan" "oatly" "pinkcan" "stapler" "styrofoam" "egg" "napkin" "cube" "bottle" "half" "milk")
# nums=("1" "1-2" "1-3" "1-4" "1-5")
# for obj in ${objs[@]}; do
#     for num in ${nums[@]}; do
#         if [ ! -d ./results/vision_${obj}/${obj}_${num}/bundlesdf_iteration_1/pll_id_04/geom_for_bsdf ]; then
#             python examples/contactnets_vision.py --run-name=04 --vision-asset=${obj}_${num} --cycle-iteration=1 --bundlesdf-id=00 --w-bsdf=0 --skip-videos=all --clear-data
#         else
# 	    echo "Already found results for ${obj}_${num}"
#     	fi
# 	if [ ! -d ./results/vision_${obj}/${obj}_${num}/bundlesdf_iteration_1/pll_id_04/geom_for_bsdf ]; then
#     	    python examples/restart_vision_run.py --run-name=04 --vision-asset=${obj}_${num} --cycle-iteration=1
# 	fi
#     done
# done


# ### Experiments 6/2 to run PLL on TagSLAM trajectories with no vision geometry supervision.
# source /mnt/data0/minghz/repos/bundlenets/dair_pll/pll_env/bin/activate
# export PYTHONPATH=${PYTHONPATH}:/mnt/data0/minghz/repos/bundlenets/dair_pll
# objs=("egg" "napkin" "cube" "bottle" "half" "milk")
# nums=("1" "1-2" "1-3") # "1-4" "1-5")
# for obj in ${objs[@]}; do
#     for num in ${nums[@]}; do
#         if [ ! -d ./results/vision_${obj}/${obj}_${num}/tagslam/pll_id_06/geom_for_bsdf ]; then
#             python examples/contactnets_vision.py --run-name=06 --vision-asset=${obj}_${num} --cycle-iteration=0 --w-bsdf=0 --skip-videos=all --clear-data
#         else
#             echo "Already found results for ${obj}_${num}"
#         fi
#         if [ ! -d ./results/vision_${obj}/${obj}_${num}/tagslam/pll_id_06/geom_for_bsdf ]; then
#             python examples/restart_vision_run.py --run-name=06 --vision-asset=${obj}_${num} --cycle-iteration=0
#         fi
#     done
# done
# nums=("1-4" "1-5")
# for obj in ${objs[@]}; do
#     for num in ${nums[@]}; do
#         if [ ! -d ./results/vision_${obj}/${obj}_${num}/tagslam/pll_id_06/geom_for_bsdf ]; then
#             python examples/contactnets_vision.py --run-name=06 --vision-asset=${obj}_${num} --cycle-iteration=0 --w-bsdf=0 --skip-videos=all --clear-data
#         else
#             echo "Already found results for ${obj}_${num}"
#         fi
#         if [ ! -d ./results/vision_${obj}/${obj}_${num}/tagslam/pll_id_06/geom_for_bsdf ]; then
#             python examples/restart_vision_run.py --run-name=06 --vision-asset=${obj}_${num} --cycle-iteration=0
#         fi
#     done
# done


# ### Experiments 6/2 to run PLL on BundleSDF trajectories with no vision geometry supervision.
# source /mnt/data0/minghz/repos/bundlenets/dair_pll/pll_env/bin/activate
# export PYTHONPATH=${PYTHONPATH}:/mnt/data0/minghz/repos/bundlenets/dair_pll
# objs=("bakingbox" "cardboard" "crushedcan" "gallon" "greencan" "oatly" "pinkcan" "stapler" "styrofoam" "egg" "napkin" "cube" "bottle" "half" "milk")
# nums=("1" "1-2" "1-3" "1-4" "1-5")
# for obj in ${objs[@]}; do
#     for num in ${nums[@]}; do
#         if [ ! -d ./results/vision_${obj}/${obj}_${num}/bundlesdf_iteration_1/pll_id_05/geom_for_bsdf ]; then
#             python examples/contactnets_vision.py --run-name=05 --vision-asset=${obj}_${num} --cycle-iteration=1 --bundlesdf-id=00 --w-bsdf=0 --skip-videos=all --clear-data
#         else
# 	    echo "Already found results for ${obj}_${num}"
#     	fi
# 	if [ ! -d ./results/vision_${obj}/${obj}_${num}/bundlesdf_iteration_1/pll_id_05/geom_for_bsdf ]; then
#     	    python examples/restart_vision_run.py --run-name=05 --vision-asset=${obj}_${num} --cycle-iteration=1
# 	fi
#     done
# done

# ### Experiments 6/2 to run PLL on BundleSDF trajectories with no vision geometry supervision.
# source /mnt/data0/minghz/repos/bundlenets/dair_pll/pll_env/bin/activate
# export PYTHONPATH=${PYTHONPATH}:/mnt/data0/minghz/repos/bundlenets/dair_pll
# objs=("bakingbox" "cardboard" "crushedcan" "gallon" "greencan" "oatly" "pinkcan" "stapler" "styrofoam" "egg" "napkin" "cube" "bottle" "half" "milk")
# nums=("1" "1-2" "1-3")
# for obj in ${objs[@]}; do
#     for num in ${nums[@]}; do
#         if [ ! -d ./results/vision_${obj}/${obj}_${num}/bundlesdf_iteration_1/pll_id_07/geom_for_bsdf ]; then
#             python examples/contactnets_vision.py --run-name=07 --vision-asset=${obj}_${num} --cycle-iteration=1 --bundlesdf-id=00 --w-bsdf=0 --skip-videos=all --clear-data
#         else
# 	    echo "Already found results for ${obj}_${num}"
#     	fi
# 	if [ ! -d ./results/vision_${obj}/${obj}_${num}/bundlesdf_iteration_1/pll_id_07/geom_for_bsdf ]; then
#     	    python examples/restart_vision_run.py --run-name=07 --vision-asset=${obj}_${num} --cycle-iteration=1
# 	fi
#     done
# done

# nums=("1-4" "1-5")
# for obj in ${objs[@]}; do
#     for num in ${nums[@]}; do
#         if [ ! -d ./results/vision_${obj}/${obj}_${num}/bundlesdf_iteration_1/pll_id_07/geom_for_bsdf ]; then
#             python examples/contactnets_vision.py --run-name=07 --vision-asset=${obj}_${num} --cycle-iteration=1 --bundlesdf-id=00 --w-bsdf=0 --skip-videos=all --clear-data
#         else
# 	    echo "Already found results for ${obj}_${num}"
#     	fi
# 	if [ ! -d ./results/vision_${obj}/${obj}_${num}/bundlesdf_iteration_1/pll_id_07/geom_for_bsdf ]; then
#     	    python examples/restart_vision_run.py --run-name=07 --vision-asset=${obj}_${num} --cycle-iteration=1
# 	fi
#     done
# done


### EXPERIMENTS 6/5 VYSICS ON MORE SINGLE TOSSES
source /mnt/data0/minghz/repos/bundlenets/dair_pll/pll_env/bin/activate
export PYTHONPATH=${PYTHONPATH}:/mnt/data0/minghz/repos/bundlenets/dair_pll
objs=("bakingbox" "cardboard" "crushedcan" "gallon" "greencan" "oatly" "pinkcan" "stapler" "styrofoam" "egg" "napkin" "cube" "bottle" "half" "milk")
nums=("2" "3" "4" "5")
for obj in ${objs[@]}; do
    for num in ${nums[@]}; do
        if [ ! -d ./results/vision_${obj}/${obj}_${num}/bundlesdf_iteration_1/pll_id_00/geom_for_bsdf ]; then
            python examples/contactnets_vision.py --run-name=07 --vision-asset=${obj}_${num} --cycle-iteration=1 --bundlesdf-id=00 --skip-videos=all
        else
	    echo "Already found results for ${obj}_${num}"
    	fi
        if [ ! -d ./results/vision_${obj}/${obj}_${num}/bundlesdf_iteration_1/pll_id_00/geom_for_bsdf ]; then
            python examples/restart_vision_run.py --run-name=00 --vision-asset=${obj}_${num} --cycle-iteration=1
        fi
    done
done


# ### Experiments 6/2 to run PLL on BundleSDF trajectories with no vision geometry supervision.
# source /mnt/data0/minghz/repos/bundlenets/dair_pll/pll_env/bin/activate
# export PYTHONPATH=${PYTHONPATH}:/mnt/data0/minghz/repos/bundlenets/dair_pll
# objs=("bakingbox" "cardboard" "crushedcan" "gallon" "greencan" "oatly" "pinkcan" "stapler" "styrofoam" "egg" "napkin" "cube" "bottle" "half" "milk")
# nums=("1" "1-2" "1-3")
# for obj in ${objs[@]}; do
#     for num in ${nums[@]}; do
#         if [ ! -d ./results/vision_${obj}/${obj}_${num}/bundlesdf_iteration_1/pll_id_07/geom_for_bsdf ]; then
#             python examples/contactnets_vision.py --run-name=07 --vision-asset=${obj}_${num} --cycle-iteration=1 --bundlesdf-id=00 --w-bsdf=0 --skip-videos=all --clear-data
#         else
# 	    echo "Already found results for ${obj}_${num}"
#     	fi
# 	if [ ! -d ./results/vision_${obj}/${obj}_${num}/bundlesdf_iteration_1/pll_id_07/geom_for_bsdf ]; then
#     	    python examples/restart_vision_run.py --run-name=07 --vision-asset=${obj}_${num} --cycle-iteration=1
# 	fi
#     done
# done

# nums=("1-4" "1-5")
# for obj in ${objs[@]}; do
#     for num in ${nums[@]}; do
#         if [ ! -d ./results/vision_${obj}/${obj}_${num}/bundlesdf_iteration_1/pll_id_07/geom_for_bsdf ]; then
#             python examples/contactnets_vision.py --run-name=07 --vision-asset=${obj}_${num} --cycle-iteration=1 --bundlesdf-id=00 --w-bsdf=0 --skip-videos=all --clear-data
#         else
# 	    echo "Already found results for ${obj}_${num}"
#     	fi
# 	if [ ! -d ./results/vision_${obj}/${obj}_${num}/bundlesdf_iteration_1/pll_id_07/geom_for_bsdf ]; then
#     	    python examples/restart_vision_run.py --run-name=07 --vision-asset=${obj}_${num} --cycle-iteration=1
# 	fi
#     done
# done
