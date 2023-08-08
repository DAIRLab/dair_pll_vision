import os
import shutil

def append_subfolders(src_folder, dst_folder, subfolder_name, verbose=False):
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

        # Create a list to store the file paths in order
        file_paths = []

        # Loop through the files in each subfolder and store the file paths
        for root, _, files in os.walk(subfolder_path):
            for file in files:
                if file.endswith(".txt") or file.endswith(".png"):
                    file_path = os.path.join(root, file)
                    file_paths.append(file_path)

        # Sort the file paths in ascending order based on the file name
        file_paths.sort(key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))

        # Loop through the sorted file paths and copy the files to the destination subfolder
        for idx, file_path in enumerate(file_paths, start=1):
            # Get the file extension to preserve the original extension
            _, ext = os.path.splitext(file_path)

            # Generate the new file name with the incremental number
            new_file_name = str(idx).zfill(4) + ext

            # Find the next available sequence number in the destination folder
            while os.path.exists(os.path.join(dst_subfolder, new_file_name)):
                idx += 1
                new_file_name = str(idx).zfill(4) + ext

            # Copy the file to the destination subfolder
            new_file_path = os.path.join(dst_subfolder, new_file_name)
            print('file_path', file_path, '-> new_file_path', new_file_path)
            shutil.copy(file_path, new_file_path)

if __name__ == "__main__":
    src_folder = "data"
    dst_folder = os.path.join(src_folder, "old_toss_all")
    subfolders_to_append = ["annotated_poses", "depth", "rgb", "masks"]

    os.makedirs(dst_folder, exist_ok=True)

    for subfolder_name in subfolders_to_append:
        append_subfolders(src_folder, dst_folder, subfolder_name, verbose=True)
