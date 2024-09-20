# kaggle competitions download -c rsna-pneumonia-detection-challenge -p rsna-pneumonia-dataset
import os
import pandas as pd
import pydicom
import numpy as np
from PIL import Image
import pydicom.uid
from pydicom.dataset import Dataset, FileDataset
import datetime

# Pfade der Datensätze
rsna_dataset_path = 'rsna-pneumonia-dataset/stage_2_train_images'
rsna_labels_path = 'rsna-pneumonia-dataset/stage_2_train_labels.csv'
rsna_detailed_class_info_path = 'rsna-pneumonia-dataset/stage_2_detailed_class_info.csv'
chest_xray_path = 'chest-xray-pneumonia-dataset/chest_xray'

# Erstellen von DICOM-Metadaten für die umgewandelten Bilder
def create_dicom_from_image(image_path, output_path, patient_id):
    img = Image.open(image_path).convert('L')  # Konvertiere das Bild in Graustufen
    pixel_array = np.array(img)

    # Erstelle ein DICOM-Dataset
    file_meta = pydicom.Dataset()
    file_meta.MediaStorageSOPClassUID = pydicom.uid.SecondaryCaptureImageStorage
    file_meta.MediaStorageSOPInstanceUID = pydicom.uid.generate_uid()
    file_meta.TransferSyntaxUID = pydicom.uid.ImplicitVRLittleEndian
    file_meta.ImplementationClassUID = pydicom.uid.generate_uid()
    
    ds = FileDataset(output_path, {}, file_meta=file_meta, preamble=b"\0" * 128)

    # Fülle die nötigen Felder des DICOM-Datensatzes
    ds.Modality = 'DX'  # Radiographie
    ds.PatientID = patient_id
    ds.PatientName = patient_id
    ds.Rows, ds.Columns = pixel_array.shape
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.BitsStored = 8
    ds.BitsAllocated = 8
    ds.HighBit = 7
    ds.PixelRepresentation = 0
    ds.PixelData = pixel_array.tobytes()

    # Erstelle notwendige UID
    ds.StudyInstanceUID = pydicom.uid.generate_uid()
    ds.SeriesInstanceUID = pydicom.uid.generate_uid()
    ds.SOPInstanceUID = pydicom.uid.generate_uid()
    ds.SOPClassUID = pydicom.uid.SecondaryCaptureImageStorage

    # Füge Datumsinformationen hinzu
    dt = datetime.datetime.now()
    ds.ContentDate = dt.strftime('%Y%m%d')
    ds.ContentTime = dt.strftime('%H%M%S')

    # Speichere als DICOM
    ds.save_as(output_path)

# Füge die neuen Bilder dem RSNA-Datensatz hinzu
def integrate_chest_xray_into_rsna():
    new_patient_ids = []
    new_labels = []
    new_class_info = []

    # Durchlaufe die Ordner des Chest-Xray-Datensatzes (train, test, val)
    for phase in ['train', 'test', 'val']:
        phase_path = os.path.join(chest_xray_path, phase)
        for label in ['NORMAL', 'PNEUMONIA']:
            label_path = os.path.join(phase_path, label)
            for img_name in os.listdir(label_path):
                if img_name.endswith(('.jpg', '.jpeg', '.png')):
                    img_path = os.path.join(label_path, img_name)

                    # Erstelle eine neue eindeutige patientId
                    patient_id = f"chestxray-{phase}-{img_name.split('.')[0]}"

                    # Umwandele das Bild in DICOM und speichere es im RSNA-Bildordner
                    dicom_output_path = os.path.join(rsna_dataset_path, f"{patient_id}.dcm")
                    create_dicom_from_image(img_path, dicom_output_path, patient_id)

                    # Füge die neuen Labels und Klassifizierungsinformationen hinzu
                    if label == 'NORMAL':
                        new_labels.append([patient_id, np.nan, np.nan, np.nan, np.nan, 0])
                        new_class_info.append([patient_id, 'Normal'])
                    else:
                        new_labels.append([patient_id, np.nan, np.nan, np.nan, np.nan, 1])
                        new_class_info.append([patient_id, 'Lung Opacity'])

                    new_patient_ids.append(patient_id)

    # Ergänze die neuen Daten in den CSV-Dateien
    rsna_labels_df = pd.read_csv(rsna_labels_path)
    rsna_class_info_df = pd.read_csv(rsna_detailed_class_info_path)

    new_labels_df = pd.DataFrame(new_labels, columns=rsna_labels_df.columns)
    new_class_info_df = pd.DataFrame(new_class_info, columns=rsna_class_info_df.columns)

    # Kombiniere die alten und neuen Daten und speichere sie
    combined_labels_df = pd.concat([rsna_labels_df, new_labels_df], ignore_index=True)
    combined_class_info_df = pd.concat([rsna_class_info_df, new_class_info_df], ignore_index=True)

    combined_labels_df.to_csv(rsna_labels_path, index=False)
    combined_class_info_df.to_csv(rsna_detailed_class_info_path, index=False)

    print(f"Insgesamt {len(new_patient_ids)} neue Bilder integriert.")

if __name__ == "__main__":
    integrate_chest_xray_into_rsna()
