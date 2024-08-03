"""Helper script to reorganize the assets folder.  This is a one-time script to
move files around to match the expected structure for the PLL assets directory
from BundleSDF results."""

import os
import os.path as op
import pdb

from dair_pll import file_utils


pdb.set_trace()

for eg_vision_cube in os.listdir(file_utils.ASSETS_DIR):
    if ('vision' not in eg_vision_cube) or \
        not os.isdir(op.join(file_utils.ASSETS_DIR, eg_vision_cube)):
        print(f'Skipping {eg_vision_cube=}')
        continue

    vision_folder = op.join(file_utils.ASSETS_DIR, eg_vision_cube)
    for subfolder in os.listdir(vision_folder):
        if 'toss' not in os.listdir(op.join(vision_folder, subfolder)):
            print(f'Skipping {subfolder=} of {eg_vision_cube=}')
            continue

        print(f'\n===== Checking {eg_vision_cube}/{subfolder} =====')
        toss_folder = op.join(vision_folder, subfolder, 'toss')
        toss_num = int(subfolder.split('_')[-1])

        # First check for tagslam.pt.
        if 'tagslam.pt' in os.listdir(toss_folder):
            print(f'\t{eg_vision_cube}/{subfolder}/tagslam.pt -> ' + \
                  f'{eg_vision_cube}/{subfolder}/tagslam/{toss_num}.pt.')
            pdb.set_trace()
            os.system(f"mkdir {op.join(toss_folder, 'tagslam')}")
            move_command = f"mv {op.join(toss_folder, 'tagslam.pt')} " + \
                f"{op.join(toss_folder, 'tagslam', f'{toss_num}.pt')}"
            os.system(move_command)

        # Second check for any BundleSDF folders.
        bsdf_folder = op.join(toss_folder, 'bundlesdf_iteration_1')
        if not op.isdir(bsdf_folder):
            print(f'No BundleSDF folder in {eg_vision_cube}/{subfolder}.')
            continue
        if 'bundlesdf_id_1.pt' not in os.listdir(bsdf_folder):
            print(f'Found BundleSDF folder in {eg_vision_cube}/{subfolder}, but no ' + \
                  f'bundlesdf_id_1.pt.')
            continue
        print(f'\t{eg_vision_cube}/{subfolder}/bundlesdf_iteration_1/bundlesdf_id_1.pt' + \
              f' -> {eg_vision_cube}/{subfolder}/bundlesdf_iteration_1/bundlesdf_id_1/{toss_num}.pt.')
        pdb.set_trace()
        os.system(f"mkdir {op.join(toss_folder, bsdf_folder, 'bundlesdf_id_1')}")
        move_command = f"mv {op.join(toss_folder, bsdf_folder, 'bundlesdf_id_1.pt')} " + \
            f"{op.join(toss_folder, bsdf_folder, 'bundlesdf_id_1', f'{toss_num}.pt')}"
        os.system(move_command)

print('\nDone.')
