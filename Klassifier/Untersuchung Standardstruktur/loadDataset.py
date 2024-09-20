
#pip install torch torchvision pydicom pandas tqdm


# mkdir .kaggle
#cd .kaggle
# nano .kaggle
# {"username":"program1","key":"3762cc3c1125f11ea18c930a0ee35763"}
# pip install kaggle

import os
import zipfile
import subprocess
import pandas as pd
from shutil import move

# Kaggle Dataset herunterladen und entpacken
def download_and_extract_dataset(dataset_name, dataset_path):
    # Erstelle den Ordner, falls er nicht existiert
    os.makedirs('~/.kaggle', exist_ok=True)

    # Stelle sicher, dass die Kaggle-Anmeldedaten vorhanden sind
    if not os.path.exists(os.path.expanduser('~/.kaggle/kaggle.json')):
        raise Exception("API-Schlüssel (kaggle.json) nicht gefunden! Stelle sicher, dass der Schlüssel unter ~/.kaggle/ liegt.")
    
    os.makedirs(dataset_path, exist_ok=True)

    print(f"Lade den {dataset_name}-Datensatz herunter...")
    try:
        # Lade den Datensatz von Kaggle herunter
        subprocess.run(['kaggle', 'datasets', 'download', '-d', dataset_name, '-p', dataset_path], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Fehler beim Herunterladen des Datensatzes {dataset_name}: {e}")
        return

    print(f"Datensatz {dataset_name} erfolgreich heruntergeladen.")
    
    # Entpacken der heruntergeladenen Dateien
    print(f"Entpacke den {dataset_name}-Datensatz...")
    for file in os.listdir(dataset_path):
        if file.endswith('.zip'):
            file_path = os.path.join(dataset_path, file)
            try:
                with zipfile.ZipFile(file_path, 'r') as zip_ref:
                    zip_ref.extractall(dataset_path)
                print(f"Entpackt: {file}")
            except zipfile.BadZipFile as e:
                print(f"Fehler beim Entpacken der Datei {file}: {e}")

def integrate_chest_xray_pneumonia(rsna_path, chest_xray_path):
    print("Integriere den Chest X-ray Pneumonia-Datensatz in den RSNA-Datensatz...")
    
    # Beispielhafte Annahme: Der RSNA-Datensatz enthält eine `labels.csv` Datei für die Labels
    labels_file = os.path.join(rsna_path, 'labels.csv')
    if not os.path.exists(labels_file):
        # Erstelle eine neue leere Label-Datei falls sie nicht existiert
        columns = ['filename', 'label']
        pd.DataFrame(columns=columns).to_csv(labels_file, index=False)

    # Lade die aktuelle Label-Datei
    labels_df = pd.read_csv(labels_file)

    # Gehe davon aus, dass der Chest X-ray Pneumonia-Datensatz in den Unterordnern 'train', 'val', 'test' liegt
    for subdir in ['train', 'val', 'test']:
        subdir_path = os.path.join(chest_xray_path, subdir)
        if not os.path.exists(subdir_path):
            continue
        
        for label in ['NORMAL', 'PNEUMONIA']:
            label_path = os.path.join(subdir_path, label)
            if not os.path.exists(label_path):
                continue
            
            for img_file in os.listdir(label_path):
                if img_file.endswith('.jpeg') or img_file.endswith('.png') or img_file.endswith('.jpg'):
                    src = os.path.join(label_path, img_file)
                    dst = os.path.join(rsna_path, 'images', img_file)
                    
                    # Bewege die Bilddateien in den RSNA-Ordner
                    move(src, dst)

                    # Füge die Label-Informationen in die `labels.csv` Datei ein
                    labels_df = labels_df.append({'filename': img_file, 'label': label}, ignore_index=True)

    # Speichere die aktualisierte Label-Datei
    labels_df.to_csv(labels_file, index=False)
    print("Integration abgeschlossen.")

if __name__ == "__main__":
    # Pfade zu den Datensätzen
    rsna_path = 'rsna-pneumonia-dataset'
    chest_xray_path = 'chest-xray-pneumonia-dataset'

    # Lade den RSNA Pneumonia Detection Challenge-Datensatz herunter
    download_and_extract_dataset('rsna-pneumonia-detection-challenge', rsna_path)

    # Lade den Chest X-ray Pneumonia-Datensatz herunter
    download_and_extract_dataset('paultimothymooney/chest-xray-pneumonia', chest_xray_path)

    # Integriere den Chest X-ray Pneumonia-Datensatz in den RSNA-Datensatz
    integrate_chest_xray_pneumonia(rsna_path, chest_xray_path)
