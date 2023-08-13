#!/bin/bash

# Running the first command seven times with old_toss_x replaced by old_toss_1, old_toss_2, ... , old_toss_7
for i in {1..10}
do
    python run_custom.py --mode run_video --video_dir ./data/bottle_10_tosses/bottle_toss_$i --out_folder ./results/bottle_10_tosses/bottle_toss_$i --use_segmenter 1 --use_gui 0 --debug_level 2 || true
done

# Running the second command seven times with bottle_x replaced by bottle_1, bottle_2, ... , bottle_7
for i in {1..10}
do
    python run_custom.py --mode global_refine --video_dir ./data/bottle_10_tosses/bottle_toss_$i --out_folder ./results/bottle_10_tosses/bottle_toss_$i --use_segmenter 1 --use_gui 0 --debug_level 2 || true
done
