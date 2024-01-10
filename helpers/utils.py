import os

def fill_gaps_in_files(folder_path):
    # List all .pt files and sort them
    files = sorted([f for f in os.listdir(folder_path) if f.endswith('.pt')])

    # Rename files to ensure a continuous sequence
    for i, file in enumerate(files, start=1):
        new_file_name = f"{i}.pt"
        if file != new_file_name:
            os.rename(os.path.join(folder_path, file), os.path.join(folder_path, new_file_name))

if __name__ == '__main__':
    path = './assets/bundlesdf_cube_test'
    fill_gaps_in_files(path)