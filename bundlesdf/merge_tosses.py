import os
import shutil

def append_subfolders(src_folder, dst_folder, subfolder_name):
    dst_subfolder = os.path.join(dst_folder, subfolder_name)
    os.makedirs(dst_subfolder, exist_ok=True)

    # Loop through the subfolders from old_toss_1 to old_toss_7 in the source directory
    for i in range(1, 8):
        subfolder = f"old_toss_{i}"
        subfolder_path = os.path.join(src_folder, subfolder, subfolder_name)

        if not os.path.exists(subfolder_path) or not os.path.isdir(subfolder_path):
            continue

        # Print subfolder_path in each iteration
        print('subfolder_path:', subfolder_path)

        # Loop through the files in each subfolder
        for root, _, files in os.walk(subfolder_path):
            for file in files:
                old_file_path = os.path.join(root, file)

                # Get the sequence number from the file name
                seq_number = int(file.split(".")[0])

                # Find the next available sequence number in the destination folder
                while os.path.exists(os.path.join(dst_subfolder, file)):
                    seq_number += 1
                    file = file.replace(str(seq_number - 1).zfill(4), str(seq_number).zfill(4))

                new_file_path = os.path.join(dst_subfolder, file)
                shutil.copy(old_file_path, new_file_path)

if __name__ == "__main__":
    src_folder = "data"
    dst_folder = os.path.join(src_folder, "old_toss_all")
    subfolders_to_append = ["annotated_poses", "depth", "rgb", "masks"]

    os.makedirs(dst_folder, exist_ok=True)

    for subfolder_name in subfolders_to_append:
        append_subfolders(src_folder, dst_folder, subfolder_name)
