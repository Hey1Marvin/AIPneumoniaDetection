# Importieren Sie die tkinter-Modul
import tkinter as tk
from tkinter import filedialog
from PIL import ImageTk, Image
from Neural import Network
import numpy as np

#Load dataset:
# Definieren Sie die Funktion load_images, um die Bilder und die Labels aus einer .npz-Datei zu laden
def load_images (filename):
  # Laden Sie die Arrays aus der Datei mit np.load
  data = np.load (filename)
  # Holen Sie sich die Arrays mit den entsprechenden Schlüsseln
  bilder = data ["bilder"]
  labels = data ["labels"]
  # Geben Sie eine Nachricht aus, um den Erfolg zu bestätigen
  print ("Die Bilder und die Labels wurden aus", filename, "geladen")
  # Geben Sie die Arrays zurück
  return bilder, labels

#Hier muss der Absolute Pfad der Trainingsdaten als 'npz' Datei Angegeben werden (Numpy Dateityp)
bilder , labels = load_images('learnset.npz')


#Erstellen des Netze aus der Gespeicherten Netzstruktur
#dat = input("absoluten Filepath der Parameterdatei(pkl) eingeben: ") #Datei in der die Netzparameter gespeichert sind

model = Network()

model.load('pneumoniaParameter77.pkl')

model.summary("Pneumonea Detection", True)

model.compile(kosten_typ="cross-entropy", optimierer_typ="adam")

imgShape = (256,256)
#Um alle Bilder im vorraus zu berechnen (experimantal)
bilder2= bilder.copy()
bilder2 = bilder2.reshape(-1,1,256,256)
#out = model.predict(bilder2, 1)

# Erstellen Sie eine Funktion, die eine Datei auswählt und ihren Namen zurückgibt
def select_file ():
    # Öffnen Sie einen Dateiauswahldialog
    file_name = filedialog.askopenfilename (title='open')
    # Geben Sie den Dateinamen zurück
    return file_name

#gibt ein zufälliges Bild aus dem Datensatz zurück
def random_file():
    ind = np.random.randint(bilder.shape[0])
    bi = bilder[ind]
    la = labels[ind]
    if la == 0:
        lab = "Gesund"
    elif la == 1: 
        lab = "Virale Pneumonie"
    else: 
        lab = "Bakterielle Pneumonie"
    file_var.set(lab+" Nr. "+str(ind))
    label_var.set(lab)
    
    predicted_var.set("")
    
    
    image = Image.fromarray(bi, 'L')
    photo = ImageTk.PhotoImage (image)
    # Zeigen Sie das Bild im Image-Widget an
    image_widget.config (image=photo, width=imgShape[1], height=imgShape[0])
    image_widget.image = photo

# Erstellen Sie eine Funktion, die eine Datei lädt und anzeigt
def load_file ():
    # Wählen Sie eine Datei aus und erhalten Sie ihren Namen
    file_name = select_file ()
    # Öffnen Sie die ausgewählte Datei als Bild
    image = Image.open(file_name)
    image = image.convert('L')
    image = image.resize(*imgShape)
    bi = np.array(image, dtype=np.uint8)
    # Erstellen Sie eine Tkinter-kompatible Fotoimage, die überall verwendet werden kann, wo Tkinter ein Bildobjekt erwartet
    photo = ImageTk.PhotoImage (image)
    # Zeigen Sie das Bild im Image-Widget an
    image_widget.config(image=photo, width=imgShape[1], height=imgShape[0])
    image_widget.image = photo
    # Zeigen Sie den Dateinamen im Label-Widget an
    file_var.set(file_name)
    
    setLabel()
    
    

# Erstellen Sie eine Funktion, die eine Vorhersage basierend auf dem Dateiinhalt macht
# Sie können diese Funktion nach Ihren Bedürfnissen ändern
def setLabel():
    # Erhalten Sie den Dateinamen aus dem Label-Widget
    file_name = file_var.get()
    # Machen Sie eine einfache Vorhersage basierend auf der Dateierweiterung
    # Sie können hier eine komplexere Logik implementieren
    if "NORMAL" in file_name:
        la = "Gesund"
    elif "PNEUMONIA" in file_name and "virus" in file_name:
        la = "Virale Pneumonie"
    elif "PNEUMONIA" in file_name and "bacteria" in file_name:
        la = "Bilddatei"
    else:
        la = "Bakterielle Pneumonie"
    # Zeigen Sie die Vorhersage im Predicted-Widget an
    label_var.set(la)

def predict_file():
    img = ImageTk.getimage(image_widget.image)

    # Konvertieren Sie das PIL-Image-Objekt in einen Array
    array = np.array(img, dtype=float)

    # Berechnen Sie den Grauwert für jeden Pixel
    gray = 0.299 * array[:, :, 0] + 0.587 * array[:, :, 1] + 0.114 * array[:, :, 2]
    input = gray.reshape(1,1,*imgShape)
    out = model.predict(input)
    print(out)
    out = out[0]
    if out == 0:
        lab = "Gesund"
    elif out == 1: 
        lab = "Virale Pneumonie"
    else: 
        lab = "Bakterielle Pneumonie"
    predicted_var.set(lab)

# Erstellen Sie ein Hauptfenster
window = tk.Tk ()
# Setzen Sie den Titel des Fensters
window.title ("Pneumonia Detection")

# Erstellen Sie eine StringVar, um den Dateinamen anzuzeigen
file_var = tk.StringVar ()
# Erstellen Sie ein Label-Widget, um den Dateinamen anzuzeigen
file_label = tk.Label (window, text= "Datei")
# Packen Sie das Label-Widget
file_label.grid(row=0, column=0)
# Erstellen Sie ein Entry-Widget mit der textvariable-Option, die auf die StringVar verweist
file_entry = tk.Entry (window, textvariable=file_var)
# Packen Sie das Entry-Widget
file_entry.grid(row=0, column=1)

# Erstellen Sie ein Image-Widget, um das Bild anzuzeigen
image_widget = tk.Label (window)
# Packen Sie das Image-Widget
# Packen Sie das Image-Widget unter den anderen Widgets mit einer maximalen Breite und Höhe
image_widget.grid(row=1, column=0, columnspan=2)

# Erstellen Sie einen Button, um die Datei zu laden
load_button = tk.Button (window, text="Load", command=load_file)
# Packen Sie den Button
load_button.grid(row=2, column=0)

# Erstellen Sie einen Button, um eine zufällige Datei aus dem Datensatz zu wählen
load_button = tk.Button (window, text="Random", command=random_file)
# Packen Sie den Button
load_button.grid(row=2, column=1)

# Erstellen Sie einen Button, um die Vorhersage auszulösen
predict_button = tk.Button (window, text="Predict", command=predict_file)
# Packen Sie den Button
predict_button.grid(row= 3, column = 0, columnspan=2)


llabel_widget = tk.Label(window, text = "Label")
llabel_widget.grid(row = 4, column = 0)
label_var = tk.StringVar()
label_widget = tk.Label(window, textvariable=label_var)
label_widget.grid(row = 4, column = 1)


lpredicted_widget = tk.Label(window, text = "Prediction")
lpredicted_widget.grid(row = 5, column = 0)
# Erstellen Sie eine StringVar, um die Vorhersage anzuzeigen
predicted_var = tk.StringVar ()
# Erstellen Sie ein Label-Widget, um die Vorhersage anzuzeigen
predicted_widget = tk.Label (window, textvariable=predicted_var)
# Packen Sie das Predicted-Widget
predicted_widget.grid(row=5, column=1)

# Starten Sie die Hauptschleife des Fensters
window.mainloop ()
