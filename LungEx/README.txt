

** Segmentierungs Netze**

- Attantion U-Net: 
    - Verwendete Module: 
        - Pytorch
        - Impemntierung Attention U-Net von https://github.com/Wimukti/CXLSeg




** Verwendeter Datensatz **

chest Xray Masks and Labels
    -von: Kaggle.com
    -Download (9,58Gb): kaggle datasets download -d nikhilpandey360/chest-xray-masks-and-labels
    -Beschreibung: 
        Der Datensatz “Chest Xray Masks and Labels” wurde entwickelt, um die Forschung im Bereich der medizinischen Bildverarbeitung zu unterstützen, insbesondere bei der Diagnose von Brustkrankheiten wie Tuberkulose (TB). Hier sind einige wichtige Details zum Datensatz:

        Ursprung und Hintergrund
        Der Datensatz wurde von Forschern der U.S. National Library of Medicine erstellt und kombiniert zwei öffentlich zugängliche Brust-Röntgen-Datensätze: den Montgomery County (MC) Set und den Shenzhen Set1. Diese Datensätze wurden entwickelt, um die Entwicklung von computergestützten Diagnosesystemen zu fördern, da es an öffentlich verfügbaren Röntgenbildern mangelte, die für das Training von maschinellen Lernalgorithmen geeignet sind1.

        Inhalt des Datensatzes
        Bilder:
        Der Datensatz enthält Brust-Röntgenbilder, die in verschiedenen Formaten wie PNG vorliegen.
        Die Bilder stammen aus zwei Hauptquellen:
        Montgomery County Set: Enthält 138 frontale Brust-Röntgenbilder, darunter 80 normale Fälle und 58 TB-Fälle1.
        Shenzhen Set: Enthält 662 frontale Brust-Röntgenbilder, darunter 326 normale Fälle und 336 TB-Fälle1.
        Masken:
        Zu jedem Röntgenbild gibt es entsprechende Segmentierungsmasken, die bestimmte Bereiche wie die Lungen markieren.
        Diese Masken wurden manuell unter der Aufsicht von Radiologen erstellt1.
        Labels:
        Die Labels enthalten Informationen über diagnostizierte Zustände oder Anomalien, die auf den Röntgenbildern zu sehen sind.
        Diese Informationen umfassen Details wie das Geschlecht und Alter des Patienten sowie beobachtete Lungenanomalien.

    - Aufbau: 
        Inhalt: CXR_png  ClinicalReadings  NLM-ChinaCXRSet-ReadMe.docx  NLM-MontgomeryCXRSet-ReadMe.pdf  masks  test
            CXR_png: Ordner der Röntgenbilder(800) im Format:  CHNCXR_0537_1.png oder MCUCXR_0048_0.png
            ClinicalReadings: Ordner mit .txt Dateien zu den Bildern im Format: CHNCXR_0622_1.txt  MCUCXR_0313_1.txt (z.B:female 49yrs \n left PTB)
            NLM-ChinaCXRSet-ReadMe.docx: Doku
            NLM-MontgomeryCXRSet-ReadMe.pdf: Doku
            masks: Ornder mit den Binären Masken (704) zu den Röntgenbildern (nicht zu allen): CHNCXR_0598_1_mask.png  MCUCXR_0266_1.png
            test: Ornder mit Testbildern (ohne Binäre Masken)

    - renameMasks.py: fügt jeder Datei aus dem Ordner masks die Endung _mask an (z.B. MCUCXR_0266_1.png fehlt diese)
    - moveUniqueFiles.py: verschiebt die Bilder aus CXR_png ohne zugehörige Segmentierungsmaske in einen seperaten Ordner
