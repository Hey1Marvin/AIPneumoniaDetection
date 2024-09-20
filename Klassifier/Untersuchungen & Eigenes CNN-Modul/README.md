# PneumoniaDetection
Im Folgenden sind die Programme zu der Seminararbeit "Pneumonia Detection mithilfe von CNNs" aufgelistet. Diese unterscheiden sich zu einen in Untersuchen der optimalen Strukur und das eigene Modul mit dem erstellten Netz. Aufgrund der Begrenzung des Speicherplatzes wurden hier die nur die wichtigsten Erungenschaften der Arbeit abgespeicher. Zum Beipiel die Datensätzte waren unmöglich hochzuladen, da uns kein Server mit der benötigten Speichergröße bereitstand. Wenn Sie mehr über unsere Arbeit erfahren wollen, so besuchen Sie doch auch unser Website unter: https://asgspez.wixsite.com/aipneumoniadetection


#Untersuchung der optimalen Struktur

- einlesen.py
    Dient zum Einlesen des Datensatzes in ein Numpy array und Speichern dieses Datensatze als 'npz' file
  
- CreateNewData.py
    Dient zum vervielfältigen des Datensatzes nimmt ebenfalls den Datensatz in Bildform
  
- Neuronal.py
    Modul zum Erstellen Neuronaler Netze
  
- ErweiterungMorePatientData.py
    Implementierung der Erweiterung mit zusätzlichen Patientendaten
  
- TestLayerNumber.py: Testen der optimalen Netzstruktur (Benötigt Datensatz als Bilder wie auf https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia/data erhältich)


#Eigenes Modul und erstellen des Netzes

- Neural
  Modul in welchem die Klassen für ein Neuronales Netz Implementiert sind (Kommentare wurden teilweise KI-Generiert nach dem Numpy Standard)

- html
  Ordner in der die Dokumentation von Neural sich befindet
  index.html = Startseite (dies würde mit pydoc generiert)

- classes - Hirarchiestruktur des Neural Modules visuell dargestellt

- PneumoniaDetection.py
  Erstellt ein Netz zur Pneumonia Detection mithilfe des eignen Moduls (Neural), trainiert dieses mithilfe con Learnset.npz und speicher die Parameter anschießend ab
  
- show ist ein Beispielprogramm, welches das gespeicherte Netz aus einer pkl Datei läd und anschliesend mit diesem Netz dann Bilder aus dem Lernset oder eigene Bilder auf Pneumonie untersuchen. (Visuelle Darstellung über Tkinter)


- Als Datensatz wird der Chest X-Ray Datensatz von Kaggle mit 5863 Bildern verwendet
