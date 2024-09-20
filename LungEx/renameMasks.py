import os

def rename_files_in_directory(directory):
    for filename in os.listdir(directory):
        if filename.startswith("MCUCXR") and filename.endswith(".png"):
            new_filename = filename.replace(".png", "_mask.png")
            os.rename(os.path.join(directory, filename), os.path.join(directory, new_filename))
            print(f'Renamed: {filename} to {new_filename}')

# Beispielverwendung
directory_path = 'dataset/masks'
rename_files_in_directory(directory_path)

