# PneuNet: Pneumonia Detection mit KI

Willkommen zum Repository **PneuNet**, einem Forschungsprojekt zur automatisierten Pneumonie-Erkennung mittels Deep Learning. Dieses Projekt untersucht verschiedene Klassifikations- und Segmentierungsmethoden auf medizinischen Bilddaten (Röntgenaufnahmen der Lunge) und umfasst dabei folende Teilbereiche.

🔗 **Zusätzliche Dateien:** [Google Drive Ordner](https://drive.google.com/drive/folders/1j2y4xAuir8jAB5wOwGD7GjGeYDEXIozN)

---

## 📁 Projektstruktur
Die Dateien und Ordner folgen der logischen Struktur unserer Forschung:

### 🩺 Klassifizierer
Untersuchung verschiedener Klassifikationsmethoden zur Detektion von Pneumonie auf Röntgenbildern.
- **Datensätze:**
  - [RSNA-Pneumonia-Detection-Challenge](https://www.kaggle.com/c/rsna-pneumonia-detection-challenge) (Teil des NIH Chest X-Ray Dataset, ~30.000 Bilder)
  - [Chest XRay (Pneumonia) Dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) von Kaggle

### 🫁 LungEx (Lung Segmentation & Classification)
Analyse von Bildsegmentierungsmethoden und deren Kombination mit Klassifikatoren.
- **Datensätze:**
  - [RSNA-Pneumonia-Detection-Challenge](https://www.kaggle.com/c/rsna-pneumonia-detection-challenge)
  - [Chest XRay Masks and Labels](https://www.kaggle.com/nikhilpandey360/chest-xray-masks-and-labels)

### 🔎 ObjectDet (Objekterkennung)
Erforschung von Methoden der Objekterkennung, einschließlich Mask R-CNN und CAM.
- **Datensatz:**
  - [RSNA-Pneumonia-Detection-Challenge](https://www.kaggle.com/c/rsna-pneumonia-detection-challenge)

### 🏥 DICOM Viewer
Integration der KI in einen DICOM-Viewer zur Analyse medizinischer Bilddaten.
- **Implementierungen:**
  - **C++ (QT)**: Vollständige Implementierung (aufgrund von Speicherbeschränkungen nicht hochgeladen, siehe Abbildungen)
  - **Python**: Testimplementierung als ausführbares Programm
- **Dateien:**
  - `dicomViewer.py` – Implementierung des DICOM-Viewers in Python
  - `new` – Neuere Version des Viewers (noch in Bearbeitung)
  - `image_PyDi.jpg` – Beispielbild des Python-Viewers
  - `image_QtDi.jpg` – Beispielbild des C++-Viewers

---

## 📄 Weitere Materialien
- **Leseprobe_PneuNet.pdf** – Eine Leseprobe unserer Forschungsarbeit zur Pneumonie-Erkennung mittels Convolutional Neural Networks.

---

## 📢 Kontakt
Falls Sie Fragen haben oder das Projekt weiterentwickeln möchten, freuen wir uns über Ihr Feedback!

📩 **Kontakt:** [Ihr Name / Teamname]  
🌍 **Projektseite:** [GitHub-Link wiederholen]

---

### 🚀 Vielen Dank für Ihr Interesse an PneuNet!




Im folgenden sind nur die wichtigsten Programme aufgeführt. Des weiteren Mussten Pogrammteile wie Netzparameter, Datensätze sowie große Anwendungen (DICOM-Viwer in C++) weggelassen werden, da diese das Speicherlimit überschreiten. 
Für meh Dateien Besuchen sie unser Gihub Page:https://github.com/Hey1Marvin/PneuNet sowie unsern Onlinespeicher: https://drive.google.com/drive/folders/1j2y4xAuir8jAB5wOwGD7GjGeYDEXIozN

Sie Finden die Dateien nach er logischen Strukturierung unserer Forschung wie folgt angeordnet







**Ordner**
- Klassifizierer:
	- Untersuchung zu den Klassifizieren für Pneumonia Detection
	- Datensätze:
		- RSNA-Pneumonia-detection-Challenge (Teil des NIH Chest X-Ray 		Dataset) mit 30.000 Bildern (https://www.kaggle.com/c/rsna-		pneumonia-detection-challenge)

		- Chest XRay (Pneumoia) Dataset von Kaggle.com (https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)

- LungEx:
	- Untersuchung zur Image Segmentierung und Segmentierer+Klassifizierer
	- Datensätze:
		- RSNA-Pneumonia-detection-Challenge (Teil des NIH Chest X-Ray 		Dataset) mit 30.000 Bildern (https://www.kaggle.com/c/rsna-		pneumonia-detection-challenge)

		- chest Xray Masks and Labels kaggle.com/nikhilpandey360/chest-xray-masks-and-labels

- ObjectDet
	- Methoden der Object detection mit Mask R-CNN und CAM
	- Datensatz:
		- RSNA-Pneumonia-detection-Challenge (Teil des NIH Chest X-Ray 		Dataset) mit 30.000 Bildern (https://www.kaggle.com/c/rsna-		pneumonia-detection-challenge)

- DicomViewer:
    - Implementeriung der KI innerhalb eines DicomViewers über QT in C++ (Export des projektes zu groß für den Upload  --> Implementierung siehe Foto)
    - Implementierung mit Python als test
    ** Dateien*
    	-dicomViewer.py (ausführbares Programm / Implementierung Des DicomViewers in Python)
    	-image_PyDi.jpg (Bild der DicomViewer in Python)
    	-new(neuere Version von dicomViwer (noch in Bearbeitung)
    	-image_QtDi.jpg (Bild des DicomViewer in C++)

**Dateien**
- Leseprobe_PneuNet.pdf - Leseprobe zu unserer Facharbeiter Pneumonia Detection mithilfe von convolutional neural networks
