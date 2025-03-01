# PneuNet: Pneumonia Detection mit KI

## 📌 Projektübersicht

**Titel:** Pneumonia Detection mithilfe von Convolutional Neural Networks  

Unser Ziel ist es, Ärzte bei der Diagnose von Pneumonie zu unterstützen, Fehlentscheidungen zu vermeiden und die Genauigkeit der Erkennung zu verbessern. Durch die Optimierung von Netzarchitekturen und Lernparametern streben wir eine höhere Genauigkeit als die durchschnittlichen 76 % der menschlichen Diagnostik an.

---

## ❓ Warum ist dieses Projekt wichtig?
Jedes Jahr erkranken Millionen Menschen weltweit an Pneumonie. 2022 verstarben allein in Deutschland 16.155 Menschen an dieser Krankheit. Sie ist die häufigste tödliche Krankenhausinfektion und eine der führenden Todesursachen in unterversorgten Regionen. Eine frühzeitige Erkennung kann lebensrettend sein.

## 🏥 Wer kann die Ergebnisse nutzen?
Unsere Forschungsergebnisse können von Ärzten und Radiologen zur schnellen und präzisen Diagnosestellung genutzt werden. Die Anwendung dient der Früherkennung und trägt zu verbesserten Behandlungsergebnissen bei.

---

## 📊 Beschreibung der Datensätze
- **NIH-Chest-Xray Dataset:** 30.000 Röntgenbilder, kategorisiert in "erkrankt" und "gesund".
- **Kaggle Chest XRay Dataset:** 6.000 Röntgenbilder, unterteilt in "gesund", "bakterielle Pneumonie" und "virale Pneumonie".
- **Lungen-Segmentationsdatensatz (Kaggle):** Unterstützt die Segmentierung der Lunge zur Verbesserung der Diagnose.

### 🔧 Datenaufbereitung
Um eine hohe Modellgenauigkeit zu gewährleisten, haben wir folgende Schritte unternommen:
- **Image Augmentation:** Erhöhung der Datenvielfalt durch Transformationen.
- **Datenfusion:** Kombination mehrerer Datensätze zur Schaffung eines robusteren Trainingssets.
- **Spezifische Nutzung:** Verschiedene Datensätze werden je nach Teilbereich (Segmentierung, Klassifizierung, Objekterkennung) eingesetzt.

---

## 🧠 Methoden & Modelle
Zur optimalen Anpassung der KI führten wir mehrere Untersuchungen durch:
- Entwicklung eines eigenen **CNN-Moduls**.
- Experimente mit verschiedenen **Klassifizierungsnetzen** (z. B. EfficientNet).
- Kombination von **Segmentierung** (Lungenbereich) und **Klassifizierung** (Pneumonie).
- Anwendung von **Erklärbarkeitsmethoden** wie Mask R-CNN und CAM.

### 📈 Evaluation & Ergebnisse
Die Modelle wurden in Trainings- und Validierungsdaten aufgeteilt. Die erzielten Genauigkeiten:
- **Klassifikatoren (z. B. EfficientNet):** >87 %
- **Segmentierer (z. B. SegFormer):** >98 %
- **Mask R-CNN (Objekterkennung):** >90 %

---

## 🏗️ Entwicklung & Anwendung
Wir haben eine KI entwickelt, die Pneumonie auf Röntgenbildern mit einer Genauigkeit von über 90 % erkennen kann. Zusätzlich arbeiten wir an einer benutzerfreundlichen Umgebung für Ärzte, in der die KI integriert ist.

### 📂 DICOM Viewer (KI-Integration)
Die KI ist in einen **DICOM-Viewer** integriert, sodass Ärzte ihre Diagnosen überprüfen und relevante Bereiche direkt markiert bekommen können.

**Anleitung zur Nutzung:**
1. Die DICOM-Viewer-Anwendung starten (Python-Implementierung).
2. Eine `.dcm` (DICOM) Datei auswählen (Beispielbilder im `images`-Ordner verfügbar).
3. Das Bild wird analysiert, die KI gibt eine Diagnose aus und markiert relevante Bereiche.

---

## 🚧 Herausforderungen & Schwachstellen
**Probleme:**
- Begrenzte Datenverfügbarkeit und -qualität beeinflussen die Modellgenauigkeit.
- Eingeschränkte Rechenressourcen erschweren die Optimierung komplexer Modelle.
- Klinische Validierung ist notwendig, um die praktische Zuverlässigkeit sicherzustellen.

**Schwachstellen:**
- Die Qualität und Diversität der Trainingsdaten stellen die größte Herausforderung dar.
- Ohne ausreichend diverse und hochqualitative Daten ist die Generalisierbarkeit des Modells limitiert.

---

## 🔮 Zukunftsausblick
**Langfristige Ziele:**
- Aufbau einer umfangreichen Datenbank mit hochauflösenden Röntgenbildern.
- Weiterentwicklung und Optimierung der CNN-Architekturen für noch präzisere Diagnosen.
- Durchführung klinischer Studien zur Validierung der KI.
- Erweiterung der Anwendung auf weitere medizinische Bildgebungsverfahren.

---

## 📂 Projektstruktur
Die Dateien sind entsprechend der logischen Struktur unserer Forschung gegliedert:

### 🩺 Klassifizierer
**Untersuchung verschiedener Klassifikationsmethoden zur Pneumonie-Erkennung.**
- **Datensätze:**
  - [RSNA-Pneumonia-Detection-Challenge](https://www.kaggle.com/c/rsna-pneumonia-detection-challenge) (~30.000 Bilder)
  - [Chest XRay (Pneumonia) Dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)

### 🫁 LungEx (Segmentierung & Klassifikation)
**Analyse der Kombination von Segmentierung und Klassifikation.**
- **Datensätze:**
  - [RSNA-Pneumonia-Detection-Challenge](https://www.kaggle.com/c/rsna-pneumonia-detection-challenge)
  - [Chest XRay Masks and Labels](https://www.kaggle.com/nikhilpandey360/chest-xray-masks-and-labels)

### 🔎 ObjectDet (Objekterkennung)
**Erforschung von Mask R-CNN und CAM für erklärbare KI.**
- **Datensatz:**
  - [RSNA-Pneumonia-Detection-Challenge](https://www.kaggle.com/c/rsna-pneumonia-detection-challenge)

### 🏥 DICOM Viewer
**Integration der KI in einen DICOM-Viewer für Ärzte.**
- **Implementierungen:**
  - **C++ (QT):** Vollständige Implementierung (nicht hochgeladen, siehe Bilder)
  - **Python:** Testimplementierung
- **Dateien:**
  - `dicomViewer.py` – Implementierung in Python
  - `new` – Neuere Version in Entwicklung
  - `image_PyDi.jpg` – Beispielbild (Python)
  - `image_QtDi.jpg` – Beispielbild (C++)

---

## 📄 Weitere Materialien
- **Leseprobe_PneuNet.pdf** – Leseprobe unserer Forschungsarbeit.

---

## 📢 Kontakt
Bei Fragen oder Interesse an einer Weiterentwicklung:  
📩 **Kontakt:** pneumonia_detection Team  
🌍 **Projektseite:** [PneuNet auf GitHub](https://github.com/Hey1Marvin/PneuNet)
**Webseite:** [PneuNet HomePage](https://asgspez.wixsite.com/aipneumoniadetection)




![Logo](images/logo.png)

# PneuNet: Präzise Pneumonie-Diagnostik mittels Künstlicher Intelligenz

## Projektübersicht
**Team:** pneumonia_detection  
**Mitglieder:** Dxstxn, KiMa  
**Titel:** Pneumonia Detection mithilfe von Convolutional Neural Networks  

PneuNet ist ein hochinnovatives Forschungsprojekt, das modernste Deep-Learning-Methoden einsetzt, um die diagnostische Genauigkeit bei der Erkennung von Pneumonie signifikant zu verbessern. Unser Ziel ist es, klinische Entscheidungsprozesse zu optimieren und diagnostische Fehler drastisch zu reduzieren.

---

## Inhaltsverzeichnis
- [Projektübersicht](#projektübersicht)
- [Motivation & Relevanz](#motivation--relevanz)
- [Technische Spezifikationen](#technische-spezifikationen)
- [Daten & Vorverarbeitung](#daten--vorverarbeitung)
- [Methodik & Evaluation](#methodik--evaluation)
- [DICOM Viewer Integration](#dicom-viewer-integration)
- [Installation & Nutzung](#installation--nutzung)
- [Mitwirkung & Lizenz](#mitwirkung--lizenz)
- [Kontakt](#kontakt)

---

## Motivation & Relevanz
Pneumonie zählt weltweit zu den führenden Infektionskrankheiten und stellt vor allem in ressourcenarmen Regionen ein erhebliches Gesundheitsrisiko dar. Unsere Forschung zielt darauf ab:

- **Fehlerminimierung:** Reduktion der diagnostischen Fehlerquote, da die manuelle Diagnose oft nur ca. 76 % Genauigkeit erreicht.
- **Effizienzsteigerung:** Beschleunigung der Diagnoseprozesse in zeitkritischen klinischen Umgebungen.
- **Transparenz:** Bereitstellung erklärbarer KI-Ergebnisse zur Unterstützung des ärztlichen Entscheidungsprozesses.

---

## Technische Spezifikationen
- **Höchste Genauigkeit:** Über 90 % diagnostische Präzision.
- **Modulare Architektur:** Klare Trennung der Module für Klassifikation, Segmentierung und Objekterkennung.
- **Erklärbare KI:** Einsatz fortschrittlicher Techniken wie Mask R-CNN und Class Activation Mapping (CAM) zur transparenten Darstellung der Entscheidungsgrundlagen.
- **DICOM Viewer Integration:** Intuitive Benutzeroberfläche zur direkten Visualisierung und Interaktion mit den Analyseergebnissen.

---

## Daten & Vorverarbeitung
### Verwendete Datensätze
- **NIH-Chest-Xray Dataset:** Enthält 30.000 Röntgenbilder, kategorisiert in „erkrankt“ und „gesund“.
- **Kaggle Chest XRay Dataset:** Umfasst 6.000 Röntgenbilder, unterteilt in „gesund“, „bakterielle Pneumonie“ und „virale Pneumonie“.
- **Lungen-Segmentationsdatensatz (Kaggle):** Unterstützt die präzise Segmentierung des Lungenbereichs, um die Diagnosegenauigkeit zu erhöhen.

### Vorverarbeitung
Um die Leistungsfähigkeit unseres Modells zu maximieren, haben wir folgende Strategien implementiert:
- **Image Augmentation:** Erweiterung der Datenvielfalt durch diverse Bildtransformationen.
- **Datenfusion:** Kombination mehrerer Datensätze zur Erstellung eines robusten Trainingsmaterials.
- **Gezielte Datensatzzuordnung:** Unterschiedliche Datensätze werden je nach Anwendungsfall (Klassifikation, Segmentierung, Objekterkennung) spezifisch aufbereitet.

---

## Methodik & Evaluation
Unsere methodische Vorgehensweise basiert auf intensiven Experimenten und modernsten Ansätzen:
- **Modellentwicklung:** Aufbau eines maßgeschneiderten CNN-Moduls und Integration fortschrittlicher Klassifikationsarchitekturen (z. B. EfficientNet).
- **Segmentierung & Erklärbarkeit:** Kombination von Lungenbereichssegmentierung (z. B. mit SegFormer) mit transparenten Entscheidungsmechanismen (Mask R-CNN, CAM).
- **Evaluation:** Strikte Aufteilung in Trainings- und Validierungsdatensätze mit folgenden Ergebnissen:
  - **Klassifikation:** >87 % Genauigkeit
  - **Segmentierung:** >98 % Genauigkeit
  - **Objekterkennung (Mask R-CNN):** >90 % Genauigkeit

---

## DICOM Viewer Integration
Die Integration der KI in einen modernen DICOM-Viewer ermöglicht eine direkte klinische Anwendung:
- **Benutzerfreundliche Oberfläche:** Ärzte können einfach `.dcm` Dateien auswählen und die Analyseergebnisse direkt abrufen.
- **Direkte Visualisierung:** Die KI analysiert das Bild, liefert eine präzise Diagnose und markiert relevante Anomalien.

![DICOM Viewer Beispiel](images/dicom_viewer_example.png)

---

## Installation & Nutzung
### Installation
1. **Repository klonen:**
   ```bash
   git clone https://github.com/Hey1Marvin/PneuNet.git
   cd PneuNet

