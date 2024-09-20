import os
import shutil

def compare_and_move_files(folder1, folder2, destination_folder):
    # Erstelle den Zielordner, falls er nicht existiert
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    # Erhalte die Liste der Dateien im ersten Ordner
    files_in_folder1 = set(os.listdir(folder1))

    # Erhalte die Liste der Dateien im zweiten Ordner ohne das "_mask" vor der Dateiendung
    files_in_folder2 = {file.replace('_mask', '') for file in os.listdir(folder2)}

    # Finde die Dateien, die nur im ersten Ordner vorhanden sind
    unique_files = files_in_folder1 - files_in_folder2

    # Verschiebe die einzigartigen Dateien in den Zielordner
    for file in unique_files:
        source_path = os.path.join(folder1, file)
        destination_path = os.path.join(destination_folder, file)
        shutil.move(source_path, destination_path)
        print(f'Moved: {file} to {destination_folder}')

# Beispielverwendung
folder1_path = 'dataset/CXR_png'
folder2_path = 'dataset/masks'
destination_folder_path = 'dataset/uniqueData'

compare_and_move_files(folder1_path, folder2_path, destination_folder_path)

