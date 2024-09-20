- Datensätze:
	- RSNA-Pneumonia-detection-Challenge (Teil des NIH Chest X-Ray 		Dataset) mit 30.000 Bildern (https://www.kaggle.com/c/rsna-		pneumonia-detection-challenge)
	- Chest XRay (Pneumoia) Dataset von Kaggle.com (https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
	
Dateien:
	- trainKlassiResNet.py : trainiert ein ResNet101 Model
	- historyResNet.txt: history des Trainings von ResNet
	
	- trainKlassiConvNeXt.py: trainiert ein ConvNeXt Model
	- historyConvNeXt.txt: History des trainings von ConvNeXt
	
	- trainKlassiDense.py : trainiert ein DenseNet
	- historyDense.txt : History des trainings von DenseNet
	
	- trainKlassiEff.py: trainiert ein EfficientNet
	- historyEff.txt : Histry des trainings von Efficient Net
	
	- loadDataset.py: läd den Chest XRay (Pneumoia) Dataset von kaggle 		herunter
	- joinDataset.py: fügt den Chest XRay (Pneumoia) Dataset in den 		RSNA-Pneumonia-detection-challenge Datensatz ein