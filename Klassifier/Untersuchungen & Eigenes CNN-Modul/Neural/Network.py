 ########### used Modules ############

# numpy for linear algebra
import numpy as np

# matplotlib for plotting the loss functions and/or accuracy
import matplotlib.pyplot as plt

# confusion matrix
from sklearn.metrics import confusion_matrix

# accuracy score
from sklearn.metrics import accuracy_score

# show progress bar
from tqdm import tqdm
from alive_progress import alive_bar

# Modules for Layer and Utility functions
from .Rectifier import *
from .Layer import *

#zum Abspeichern der Parameter
import pickle


##### Classes ######


# #### [Batch Normalization class]

class BatchNorm:
    
    def __init__(self, beta=0.9, eta=1e-6, gamma = None, avg = None, std = None, alpha = None):
        '''
        Der Initialisierer der BatchReg-Klasse. Setzt das Beta und Eta.
        
        Parameter:
        beta: Beta für den moving average
        eta: 𝜂, Kleiner Float, der zur Varianz addiert wird, um Nullteiler zu vermeiden
        '''
        self.eta = eta
        self.beta = beta
        
        if gamma is not None: self.gamma = gamma
        if avg is not None: self.avg = avg
        if std is not None: self.std = std
        if alpha is not None: self.alpha = alpha

    def initialize_parameters(self, m):
        '''
        Diese Funktion legt die Parameter der Batch-Normalisierungs-Schicht fest.
        
        Parameter:
        m: Shape des Inputs zur BN-Schicht
        '''
        self.gamma = np.ones((m))
        self.alpha, self.avg, self.std  = np.zeros((m)), np.zeros((m)), np.zeros((m))

    def forward(self, x, mode='train'):
        '''
        Diese Funktion macht die Forward-Propagation. Sie rechnet den Mittelwert und die Varianz des Inputs aus, 
        normalisiert den Input und skaliert und verschiebt ihn dann.
        
        Parameter:
        x: Input zur BN-Schicht
        mode: Forward-Pass, der für das Training oder den Test benutzt wird
        
        Output:
        y: Die normalisierten, skalierten und verschobenen Input-Daten. Dieser Output wird zur nächsten Schicht im Netzwerk weitergegeben.
        '''
        if mode=='train':
            self.s, *self.m = x.shape
            self.mu = np.mean(x, axis = 0) # 𝜇
            self.var = np.var(x, axis=0) # 𝜎^2
            self.xmu = x - self.mu # x - 𝜇
            self.ivar = 1 / np.sqrt(self.var + self.eta) # 𝜎𝑖𝑛𝑣
            self.xhat = self.xmu * self.ivar
            y = self.gamma*self.xhat + self.alpha # yl
            self.avg = self.beta * self.avg + (1 - self.beta) * self.mu
            self.std = self.beta * self.std + (1 - self.beta) * self.var
        elif mode=='test':
            y = self.gamma * ((x - self.avg) / np.sqrt(self.std + self.eta)) + self.alpha
        else:
            raise ValueError('Ungültiger Forward-Batchnorm-Modus "%s"' % mode)
        return y

    def backpropagation(self, dy):
        '''
        Diese Funktion macht die Backward-Propagation. Sie rechnet die Gradienten der Skalierungs- und Verschiebungsparameter 
        und den Gradienten des Inputs aus.
        
        Parameter:
        dy: Gradient des Outputs
        
        Output:
        dx: Der Gradient des Inputs zur BN-Schicht. Dieser Gradient wird zur vorherigen Schicht im Netzwerk zurückgegeben und zum Updaten der Gewichte und des Bias in dieser Schicht benutzt.
        '''
        self.dgamma = np.sum(dy * self.xhat, axis=0)
        self.dalpha = np.sum(dy, axis=0)
        dxhat = dy * self.gamma
        dvar = np.sum(dxhat * self.xmu * (-.5) * (self.ivar**3), axis=0)
        dmu = np.sum(dxhat * (-self.ivar), axis=0)
        dx = dxhat * self.ivar + dvar * (2/self.s) * self.xmu + (1/self.s)*dmu
        return dx

    def update(self, learnrate, size, k):
        '''
        Diese Funktion updatet die Skalierungs- und Verschiebungsparameter basierend auf den während der Backward-Propagation 
        ausgerechneten Gradienten.
        
        Parameter:
        learnrate: Lernrate
        size: Batch-Size (Anzahl der Samples im Batch)
        k: Iterationsnummer
        '''
        self.gamma -= self.dgamma*(learnrate/size)
        self.alpha -= self.dalpha*(learnrate/size)
        
        


# #### Network
#Build an Neural Network    
    
class Network:
    
    def __init__(self, layers=None):
        '''
        Erzeugt ein sequentielles CNN-Modell.

        Parameters
        ----------
        layers : list, optional
            Eine Liste von Schichten, die dem Modell hinzugefügt werden sollen.
            Wenn None, wird eine leere Liste erstellt. Der Standardwert ist None.
        '''
        if layers is None:
            self.network_structure = []
        else:
            self.network_structure = layers
        self.architecture_called = False # Ein Attribut, das angibt, ob die Architektur des Modells berechnet wurde

    def add(self, schicht):
        '''
        Fügt eine Schicht zum Modell hinzu.

        Parameters
        ----------
        schicht : object
            Ein Objekt, das eine Schicht repräsentiert, z.B. Conv2D, Dense, etc.
        '''
        # Fügt die Schicht zur Liste der Schichten hinzu
        self.network_structure.append(schicht)

    def Input(self, input_shape):
        '''
        Definiert die Eingabeform des Modells.

        Parameters
        ----------
        input_shape : tuple
            Ein Tupel, das die Form der Eingabedaten angibt, z.B. (3, 32, 32) für RGB-Bilder mit 32x32 Pixeln.
        '''
        self.input_shape = input_shape # Die Dimension der Eingabe
        self.layer_out_shape = [self.input_shape] # Eine Liste, die die Ausgabeform jeder Schicht speichert
        self.schicht_name = ["Input"] # Eine Liste, die die Namen jeder Schicht speichert

    def network_architecture(self):
        '''
        Berechnet die Architektur des Modells basierend auf den hinzugefügten Schichten.
        '''
        for schicht in self.network_structure: # Iteriert über jede Schicht in der Liste
            if isinstance(schicht, Conv2D): # Wenn die Schicht eine Conv2D-Schicht ist
                if schicht.input_shape_x is not None: # Wenn die Schicht eine Eingabeform definiert hat
                    self.Input(schicht.input_shape_x) # Ruft die Input-Methode mit dieser Form auf
                schicht.get_dimensions(self.layer_out_shape[-1]) # Berechnet die Ausgabeform der Schicht basierend auf der vorherigen Schicht
                self.layer_out_shape.append(schicht.output_shape) # Fügt die Ausgabeform zur Architekturliste hinzu
                self.schicht_name.append(schicht.__class__.__name__) # Fügt den Namen der Schicht zur Namensliste hinzu
            elif isinstance(schicht, (Flatten, Pooling2D)): # Wenn die Schicht eine Flatten- oder Pooling2D-Schicht ist
                schicht.get_dimensions(self.layer_out_shape[-1]) # Berechnet die Ausgabeform der Schicht basierend auf der vorherigen Schicht
                self.layer_out_shape.append(schicht.output_shape) # Fügt die Ausgabeform zur Architekturliste hinzu
                self.schicht_name.append(schicht.__class__.__name__) # Fügt den Namen der Schicht zur Namensliste hinzu
            elif isinstance(schicht, Dense): # Wenn die Schicht eine Dense-Schicht ist
                self.layer_out_shape.append(schicht.neurons) # Fügt die Anzahl der Neuronen zur Architekturliste hinzu
                self.schicht_name.append(schicht.__class__.__name__) # Fügt den Namen der Schicht zur Namensliste hinzu
            else: # Wenn die Schicht eine andere Art von Schicht ist
                self.layer_out_shape.append(self.layer_out_shape[-1]) # Fügt die gleiche Ausgabeform wie die vorherige Schicht zur Architekturliste hinzu
                self.schicht_name.append(schicht.__class__.__name__) # Fügt den Namen der Schicht zur Namensliste hinzu

        self.network_structure = list(filter(None, self.network_structure)) # Entfernt alle None-Elemente aus der Schichtenliste
        try:
            idx = self.schicht_name.index("NoneType") # Sucht nach dem Index eines NoneType-Elements in der Namensliste
            del self.schicht_name[idx] # Löscht das Element an diesem Index aus der Namensliste
            del self.layer_out_shape[idx] # Löscht das Element an diesem Index aus der Architekturliste
        except:
            pass
        
    def summary(self, name = "Network"):
           
        if self.architecture_called==False: # Wenn die Architektur des Modells noch nicht berechnet wurde
            self.network_architecture() # Ruft die Methode network_architecture auf, um die Architektur zu berechnen
            self.architecture_called = True # Setzt das Attribut architecture_called auf True
        len_zugewiesen = [45, 26, 15] # Eine Liste von Längen, die für die Spalten der Zusammenfassung zugewiesen werden
        anzahl = {'Dense': 1, 'Activation': 1, 'Input': 1,
                'BatchNorm': 1, 'Dropout': 1, 'Conv2D': 1,
                'Pooling2D': 1, 'Flatten': 1} # Ein Wörterbuch, das die Anzahl jeder Schichtart speichert

        column = ['Layer (type)', 'Output Shape', '# of Parameters'] # Eine Liste von Spaltennamen für die Zusammenfassung

        print("Model: ", name) # Druckt den Namen des Modells
        print('-'*sum(len_zugewiesen)) # Druckt eine Trennlinie
        
        text = '' # Initialisiert einen leeren Text
        for i in range(3): # Iteriert über die drei Spalten
            text += column[i] + ' '*(len_zugewiesen[i]-len(column[i])) # Fügt den Spaltennamen und die erforderlichen Leerzeichen zum Text hinzu
        print(text) # Druckt den Text

        print('#'*sum(len_zugewiesen)) # Druckt eine Trennlinie

        gesamt_params = 0 # Initialisiert die Gesamtzahl der Parameter auf 0
        trainierbar_params = 0 # Initialisiert die Anzahl der trainierbaren Parameter auf 0
        nicht_trainierbar_params = 0 # Initialisiert die Anzahl der nicht trainierbaren Parameter auf 0

        for i in range(len(self.schicht_name)): # Iteriert über jede Schicht in der Namensliste
            # layer name
            schicht_name = self.schicht_name[i] # Speichert den Namen der Schicht
            name = schicht_name.lower() + '_' + str(anzahl[schicht_name]) + ' ' + '(' + schicht_name + ')' # Erstellt einen eindeutigen Namen für die Schicht mit ihrer Nummer und ihrem Typ
            anzahl[schicht_name] += 1 # Erhöht die Anzahl dieser Schichtart um 1

            # output shape
            try: # Versucht, die Ausgabeform der Schicht als Tupel zu erstellen
                out = '(None, ' # Beginnt das Tupel mit None für die Batch-Dimension
                for n in range(len(self.layer_out_shape[i])-1): # Iteriert über die restlichen Dimensionen außer der letzten
                    out += str(self.layer_out_shape[i][n]) + ', ' # Fügt die Dimension und ein Komma zum Tupel hinzu
                out += str(self.layer_out_shape[i][-1]) + ')'
            except: # Wenn die Ausgabeform keine Tupel ist
                out = '(None, ' + str(self.layer_out_shape[i]) + ')' 

            # number of params
            if schicht_name=='Dense': 
                h0 = self.layer_out_shape[i-1] 
                h1 = self.layer_out_shape[i]
                if self.network_structure[i-1].use_bias: 
                    params = h0*h1 + h1 
                else: 
                    params = h0*h1 
                gesamt_params += params 
                trainierbar_params += params
            elif schicht_name=='BatchNorm': # Wenn die Schicht eine BatchNormalization-Schicht ist
                h = self.layer_out_shape[i] # Speichert die Anzahl der Merkmale
                if isinstance(h, tuple): h = np.prod(h)
                params = 4*h # Berechnet die Anzahl der Parameter als das Vierfache der Merkmale
                trainierbar_params += 2*h # Addiert die Hälfte der Parameter zur Anzahl der trainierbaren Parameter hinzu
                nicht_trainierbar_params += 2*h # Addiert die Hälfte der Parameter zur Anzahl der nicht trainierbaren Parameter hinzu
                gesamt_params += params # Addiert die Anzahl der Parameter zur Gesamtzahl hinzu
            elif schicht_name=='Conv2D': # Wenn die Schicht eine Conv2D-Schicht ist
                layer = self.network_structure[i-1] # Speichert die Schicht als ein Objekt
                if layer.use_bias: # Wenn die Schicht einen Bias-Vektor verwendet
                    add_b = 1 # Speichert eine zusätzliche Einheit für den Bias
                else: # Wenn die Schicht keinen Bias-Vektor verwendet
                    add_b = 0 # Speichert keine zusätzliche Einheit für den Bias
                params = ((layer.inputC * layer.kernelH * layer.kernelW) + add_b) * layer.F # Berechnet die Anzahl der Parameter als das Produkt der Eingangskanäle, der Kernelhöhe, der Kernelbreite und der Anzahl der Filter plus die zusätzliche Einheit für den Bias
                trainierbar_params += params # Addiert die Anzahl der Parameter zur Anzahl der trainierbaren Parameter hinzu
                gesamt_params += params # Addiert die Anzahl der Parameter zur Gesamtzahl hinzu
            else: 
                
                params = 0 
            namen = [name, out, str(params)] 
            # print this row
            text = '' # Initialisiert einen leeren Text
            for j in range(3): # Iteriert über die drei Spalten
                text += namen[j] + ' '*(len_zugewiesen[j]-len(namen[j])) 
            print(text) # Druckt den Text
            if i!=(len(self.schicht_name)-1):
                print('.'*sum(len_zugewiesen))
            else:
                print('#'*sum(len_zugewiesen))
        
        print("Trainable params:", trainierbar_params) 
        print("Non-trainable params:", nicht_trainierbar_params) # Druckt die Anzahl der nicht trainierbaren Parameter
        print("Total params:", gesamt_params)
        print('-'*sum(len_zugewiesen)) # Druckt eine Trennlinie
    
    def compile(self, kosten_typ, optimierer_typ):
        '''
        Fertigt das Modell mit einer Kostenfunktion und einem Optimierer an.

        Parameters
        ----------
        self : object
            Eine Instanz der Klasse CNN.
        kosten_typ : str
            Der Name der Kostenfunktion, die für das Modell verwendet werden soll, z.B. "cross-entropy" oder "mse".
        optimierer_typ : str
            Der Name des Optimierers, der für das Modell verwendet werden soll, z.B. "sgd" oder "adam".

        Returns
        -------
        None
            Die Methode gibt nichts zurück, sondern setzt die Attribute kosten, kosten_typ und optimierer_typ der Instanz.
        '''
        self.kosten = Cost(kosten_typ) # Erstellt ein Objekt der Klasse Cost mit der angegebenen Kostenfunktion
        self.kosten_typ = kosten_typ # Speichert den Namen der Kostenfunktion als Attribut
        self.optimierer_typ = optimierer_typ # Speichert den Namen des Optimierers als Attribut

    def init_params(self):
        '''
        Initiiert die Parameter des Modells basierend auf den hinzugefügten Schichten.

        Parameters
        ----------
        self : object
            Eine Instanz der Klasse CNN.

        Returns
        -------
        None
            Die Methode gibt nichts zurück, sondern setzt die Parameter der Schichten als Attribute der Instanz.
        '''
        if not self.architecture_called: # Wenn die Architektur des Modells noch nicht berechnet wurde
            self.network_architecture() # Ruft die Methode network_architecture auf, um die Architektur zu berechnen
            self.architecture_called = True # Setzt das Attribut architecture_called auf True
        for i, schicht in enumerate(self.network_structure): # Iteriert über jede Schicht in der Liste der Schichten
            if isinstance(schicht, (Dense, Conv2D)): # Wenn die Schicht eine Dense- oder Conv2D-Schicht ist
                #print("Schicht: ", schicht.__class__.__name__, " input: ", self.architektur[i])
                schicht.initialize_parameters(self.layer_out_shape[i], self.optimierer_typ) # Ruft die Methode init_params der Schicht auf, um die Parameter zu initiieren
            elif isinstance(schicht, BatchNorm): # Wenn die Schicht eine BatchNormalization-Schicht ist
                schicht.initialize_parameters(self.layer_out_shape[i]) # Ruft die Methode init_params der Schicht auf, um die Parameter zu initiieren

    def fit(self, X, y, epochs=10, batch_size=5, learnrate=1, X_val=None, y_val=None, verbose=1, learnrate_decay=None, **kwargs):
        '''
        Trainiert das Modell mit den gegebenen Trainingsdaten und optionalen Validierungsdaten.

        Parameters
        ----------
        self : object
            Eine Instanz der Klasse CNN.
        X : array-like
            Die Eingabedaten für das Training, z.B. ein Numpy-Array oder eine Liste von Arrays.
        y : array-like
            Die Ausgabedaten für das Training, z.B. ein Numpy-Array oder eine Liste von Arrays.
        epochs : int, optional
            Die Anzahl der Epochen, die das Modell trainieren soll. Der Standardwert ist 10.
        batch_size : int, optional
            Die Größe der Minibatches, die für das Training verwendet werden sollen. Der Standardwert ist 5.
        learnrate : float, optional
            Die Lernrate, die für den Optimierer verwendet werden soll. Der Standardwert ist 1.
        X_val : array-like, optional
            Die Eingabedaten für die Validierung, z.B. ein Numpy-Array oder eine Liste von Arrays. Wenn None, wird keine Validierung durchgeführt. Der Standardwert ist None.
        y_val : array-like, optional
            Die Ausgabedaten für die Validierung, z.B. ein Numpy-Array oder eine Liste von Arrays. Wenn None, wird keine Validierung durchgeführt. Der Standardwert ist None.
        verbose : int, optional
            Ein Schalter, der angibt, ob die Trainings- und Validierungsergebnisse nach jeder Epoche gedruckt werden sollen. Wenn 1, werden die Ergebnisse gedruckt. Wenn 0, werden die Ergebnisse nicht gedruckt. Der Standardwert ist 1.
        learnrate_decay : function, optional
            Eine Funktion, die die Lernrate nach jeder Iteration anpasst. Wenn None, wird keine Lernratenanpassung durchgeführt. Der Standardwert ist None.
        **kwargs : dict, optional
            Zusätzliche Schlüsselwortargumente, die an die Funktion learnrate_decay übergeben werden sollen.

        Returns
        -------
        None
            Die Methode gibt nichts zurück, sondern aktualisiert die Parameter des Modells und speichert die Trainings- und Validierungshistorie als Attribute der Instanz.
        '''
        self.historie = {'Training Loss': [],'Validation Loss': [], 'Training Accuracy': [],  'Validation Accuracy': []} # Erstellt ein Wörterbuch, das die Trainings- und Validierungshistorie speichert
        iterationen = 0 # Initialisiert die Anzahl der Iterationen auf 0
        self.batch = batch_size # Speichert die Größe der Minibatches als Attribut
        self.init_params() # Ruft die Methode init_params auf, um die Parameter des Modells zu initiieren
        gesamt_num_batches = np.ceil(len(X)/batch_size) # Berechnet die Gesamtzahl der Minibatches

        for epoch in range(epochs): # Iteriert über jede Epoche
            kosten_train = 0 # Initialisiert die Trainingskosten auf 0
            num_batches = 0 # Initialisiert die Anzahl der Minibatches auf 0
            y_pred_train = [] # Initialisiert eine Liste, die die Vorhersagen des Modells für die Trainingsdaten speichert
            y_train = [] # Initialisiert eine Liste, die die tatsächlichen Ausgaben für die Trainingsdaten speichert

            print(f'\nEpoch: {epoch+1}/{epochs}') # Druckt die aktuelle Epoche
            with alive_bar(len(range(0, len(X), batch_size))) as bar:
                for i in range(0, len(X), batch_size): # Iteriert über jede Minibatch
                    X_batch = X[i:i+batch_size] # Extrahiert die Eingabedaten für die Minibatch
                    y_batch = y[i:i+batch_size] # Extrahiert die Ausgabedaten für die Minibatch

                    Z = X_batch.copy() # Erstellt eine Kopie der Eingabedaten

                    # feed-forward
                    for schicht in self.network_structure: # Iteriert über jede Schicht im Modell
                        Z = schicht.forward(Z) # Ruft die Methode forward der Schicht auf, um die Ausgabe der Schicht zu berechnen
                    
                    # Trainingsgenauigkeit berechnen
                    if self.kosten_typ=='cross-entropy': # Wenn die Kostenfunktion die Kreuzentropie ist
                        y_pred_train += np.argmax(Z, axis=1).tolist() # Fügt die Vorhersagen des Modells für die Minibatch zur Liste der Vorhersagen hinzu
                        y_train += np.argmax(y_batch, axis=1).tolist() # Fügt die tatsächlichen Ausgaben für die Minibatch zur Liste der Ausgaben hinzu

                    # Kosten berechnen
                    kosten_train += self.kosten.get_cost(Z, y_batch) / self.batch # Berechnet die Kosten für die Minibatch und addiert sie zu den Trainingskosten

                    # dL/daL berechnen (letzter Schicht Rückpropagationsfehler)
                    dZ = self.kosten.get_d_cost(Z, y_batch) # Berechnet den Fehler der letzten Schicht
                    # Rückpropagation
                    for schicht in self.network_structure[::-1]: # Iteriert über jede Schicht im Modell in umgekehrter Reihenfolge
                        dZ = schicht.backpropagation(dZ) # Ruft die Methode rückpropagation der Schicht auf, um den Fehler an die vorherige Schicht weiterzugeben

                    # Parameter aktualisieren
                    for schicht in self.network_structure: # Iteriert über jede Schicht im Modell
                        if isinstance(schicht, (Dense, BatchNorm, Conv2D)): # Wenn die Schicht eine Dense-, BatchNormalization- oder Conv2D-Schicht ist
                            schicht.update(learnrate, self.batch, iterationen) # Ruft die Methode aktualisieren der Schicht auf, um die Parameter der Schicht zu aktualisieren

                    # Lernratenzerfall
                    if learnrate_decay is not None: # Wenn eine Lernratenanpassungsfunktion angegeben ist
                        learnrate = learnrate_decay(iterationen, **kwargs) # Ruft die Funktion learnrate_decay auf, um die Lernrate anzupassen

                    num_batches += 1 # Erhöht die Anzahl der Minibatches um 1
                    iterationen += 1 # Erhöht die Anzahl der Iterationen um 1
                    
                    #update progress bar
                    bar()
                    
            kosten_train /= num_batches # Berechnet den Durchschnitt der Trainingskosten für die Epoche

            # Nur zum Drucken (Trainingsgenauigkeit, Validierungskosten und -genauigkeit)

            text  = f'Trainingskosten: {round(kosten_train, 4)} - ' # Erstellt einen Text, der die Trainingskosten enthält
            self.historie['Training Loss'].append(kosten_train) # Fügt die Trainingskosten zur Historie hinzu

            # Trainingsgenauigkeit

            if self.kosten_typ=='cross-entropy': # Wenn die Kostenfunktion die Kreuzentropie ist
                genauigkeit_train = np.sum(np.array(y_pred_train) == np.array(y_train)) / len(y_train) # Berechnet die Trainingsgenauigkeit für die Epoche
                text += f'Trainingsgenauigkeit: {round(genauigkeit_train, 4)}' # Fügt die Trainingsgenauigkeit zum Text hinzu
                self.historie['Training Accuracy'].append(genauigkeit_train) # Fügt die Trainingsgenauigkeit zur Historie hinzu
            else: # Wenn die Kostenfunktion eine andere ist
                text += f'Trainingsgenauigkeit: {round(kosten_train, 4)}' # Fügt die Trainingskosten als Genauigkeit zum Text hinzu
                self.historie['Training Accuracy'].append(kosten_train) # Fügt die Trainingskosten als Genauigkeit zur Historie hinzu

            if X_val is not None: # Wenn Validierungsdaten angegeben sind
                kosten_val, genauigkeit_val = self.evaluate(X_val, y_val, batch_size) # Ruft die Methode bewerten auf, um die Validierungskosten und -genauigkeit zu berechnen
                text += f' - Validierungskosten: {round(kosten_val, 4)} - ' 
                self.historie['Validation Loss'].append(kosten_val) 
                text += f'Validierungsgenauigkeit: {round(genauigkeit_val, 4)}' 
                self.historie['Validation Accuracy'].append(genauigkeit_val) # Fügt die Validierungsgenauigkeit zur Historie hinzu

            if verbose:
                    print(text)
            else:
                print()
    
    def evaluate(self, X, y, batch_size=None):
        '''
        Testet das Modell mit den gegebenen Testdaten.

        Parameters
        ----------
        self : object
            Eine Instanz der Klasse CNN.
        X : array-like
            Die Eingabedaten für den Test, z.B. ein Numpy-Array oder eine Liste von Arrays.
        y : array-like
            Die Ausgabedaten für den Test, z.B. ein Numpy-Array oder eine Liste von Arrays.
        batch_size : int, optional
            Die Größe der Minibatches, die für den Test verwendet werden sollen. Wenn None, wird die Länge von X verwendet. Der Standardwert ist None.

        Returns
        -------
        kosten : float
            Die Kosten des Modells für die Testdaten, berechnet mit der Kostenfunktion des Modells.
        genauigkeit : float
            Die Genauigkeit des Modells für die Testdaten, berechnet als der Anteil der korrekten Vorhersagen.
        '''
        if batch_size is None: # Wenn keine Batch-Größe angegeben ist
            batch_size = len(X) # Verwendet die Länge von X als Batch-Größe

        kosten = 0 # Initialisiert die Kosten auf 0
        richtig = 0 # Initialisiert die Anzahl der richtigen Vorhersagen auf 0
        num_batches = 0 # Initialisiert die Anzahl der Minibatches auf 0
        hilfe = Utility() # Erstellt ein Objekt der Klasse Utility
        Y_1hot, _ = hilfe.onehot(y) # Wandelt die Ausgabedaten in One-Hot-Vektoren um

        for i in range(0, len(X), batch_size): # Iteriert über jede Minibatch
            X_batch = X[i:i+batch_size] # Extrahiert die Eingabedaten für die Minibatch
            y_batch = y[i:i+batch_size] # Extrahiert die Ausgabedaten für die Minibatch
            Y_1hot_batch = Y_1hot[i:i+batch_size] # Extrahiert die One-Hot-Vektoren für die Minibatch
            Z = X_batch.copy() # Erstellt eine Kopie der Eingabedaten
            for schicht in self.network_structure: # Iteriert über jede Schicht im Modell
                if schicht.__class__.__name__=='BatchNorm': # Wenn die Schicht eine BatchNormalization-Schicht ist
                    Z = schicht.forward(Z, mode='test') # Ruft die Methode forward der Schicht auf, um die Ausgabe der Schicht im Testmodus zu berechnen
                else: # Wenn die Schicht eine andere Art von Schicht ist
                    Z = schicht.forward(Z) # Ruft die Methode forward der Schicht auf, um die Ausgabe der Schicht zu berechnen
            if self.kosten_typ=='cross-entropy': # Wenn die Kostenfunktion die Kreuzentropie ist
                kosten += self.kosten.get_cost(Z, Y_1hot_batch) / len(y_batch) # Berechnet die Kosten für die Minibatch und addiert sie zu den Gesamtkosten
                y_pred = np.argmax(Z, axis=1).tolist() # Berechnet die Vorhersagen des Modells für die Minibatch
                richtig += np.sum(y_pred == y_batch) # Zählt die Anzahl der richtigen Vorhersagen für die Minibatch
            else: # Wenn die Kostenfunktion eine andere ist
                kosten += self.kosten.get_cost(Z, y_batch) / len(y_batch) # Berechnet die Kosten für die Minibatch und addiert sie zu den Gesamtkosten

            num_batches += 1 # Erhöht die Anzahl der Minibatches um 1

        if self.kosten_typ=='cross-entropy': # Wenn die Kostenfunktion die Kreuzentropie ist
            genauigkeit = richtig / len(y) # Berechnet die Genauigkeit des Modells für die Testdaten
            kosten /= num_batches # Berechnet den Durchschnitt der Kosten für die Testdaten
            return kosten, genauigkeit # Gibt die Kosten und die Genauigkeit zurück
        else: # Wenn die Kostenfunktion eine andere ist
            kosten /= num_batches # Berechnet den Durchschnitt der Kosten für die Testdaten
            return kosten, kosten # Gibt die Kosten zweimal zurück

    def loss_plot(self):
        '''
        Zeigt einen Plot der Trainings- und Validierungskosten pro Epoche an.

        Parameters
        ----------
        self : object
            Eine Instanz der Klasse CNN.

        Returns
        -------
        None
            Die Methode gibt nichts zurück, sondern zeigt den Plot auf dem Bildschirm an.
        '''
        plt.plot(self.historie['Training Loss'], 'k') # Plottet die Trainingskosten in schwarz
        if len(self.historie['Validation Loss'])>0: # Wenn es Validierungskosten gibt
            plt.plot(self.historie['Validation Loss'], 'r') # Plottet die Validierungskosten in rot
            plt.legend(['Training', 'Validierung'], loc='upper right') # Fügt eine Legende mit den Namen der Kurven hinzu
            plt.title('Modellkosten') # Fügt einen Titel für den Plot hinzu
        else: # Wenn es keine Validierungskosten gibt
            plt.title('Trainingskosten') # Fügt einen Titel für den Plot hinzu
        plt.ylabel('Kosten') # Fügt eine Beschriftung für die y-Achse hinzu
        plt.xlabel('Epoche') # Fügt eine Beschriftung für die x-Achse hinzu
        plt.show() # Zeigt den Plot an

    def accuracy_plot(self):
        '''
        Zeigt einen Plot der Trainings- und Validierungsgenauigkeit pro Epoche an.

        Parameters
        ----------
        self : object
            Eine Instanz der Klasse CNN.

        Returns
        -------
        None
            Die Methode gibt nichts zurück, sondern zeigt den Plot auf dem Bildschirm an.
        '''
        plt.plot(self.historie['Training Accuracy'], 'k') # Plottet die Trainingsgenauigkeit in schwarz
        if len(self.historie['Validation Accuracy'])>0: # Wenn es Validierungsgenauigkeit gibt
            plt.plot(self.historie['Validation Accuracy'], 'r') # Plottet die Validierungsgenauigkeit in rot
            plt.legend(['Training', 'Validierung'], loc='lower right') # Fügt eine Legende mit den Namen der Kurven hinzu
            plt.title('Modellgenauigkeit') # Fügt einen Titel für den Plot hinzu
        else: # Wenn es keine Validierungsgenauigkeit gibt
            plt.title('Trainingsgenauigkeit') # Fügt einen Titel für den Plot hinzu
        plt.ylabel('Genauigkeit') # Fügt eine Beschriftung für die y-Achse hinzu
        plt.xlabel('Epoche') # Fügt eine Beschriftung für die x-Achse hinzu
        plt.show() # Zeigt den Plot an

    def predict(self, X, batch_size=None):
        '''
        Erzeugt Vorhersagen des Modells für die gegebenen Eingabedaten.

        Parameters
        ----------
        self : object
            Eine Instanz der Klasse CNN.
        X : array-like
            Die Eingabedaten, für die das Modell Vorhersagen machen soll, z.B. ein Numpy-Array oder eine Liste von Arrays.
        batch_size : int, optional
            Die Größe der Minibatches, die für die Vorhersage verwendet werden sollen. Wenn None, wird die Länge von X verwendet. Der Standardwert ist None.

        Returns
        -------
        y_pred : array-like
            Die Vorhersagen des Modells für die Eingabedaten, z.B. ein Numpy-Array oder eine Liste von Arrays.
        '''
        if batch_size is None: # Wenn keine Batch-Größe angegeben ist
            if len(X.shape) <3: 
                batch_size = 1
                le = 1
            else: 
                batch_size = X.shape[0] # Verwendet die Länge von X als Batch-Größe
                le = X.shape[0]
        else: le = X.shape[0]
        for i in range(0, le, batch_size): # Iteriert über jede Minibatch
            X_batch = X[i:i+batch_size] # Extrahiert die Eingabedaten für die Minibatch
            Z = X_batch.copy() # Erstellt eine Kopie der Eingabedaten
            for schicht in self.network_structure: # Iteriert über jede Schicht im Modell
                if schicht.__class__.__name__=='BatchNorm': # Wenn die Schicht eine BatchNormalization-Schicht ist
                    Z = schicht.forward(Z, mode='test') # Ruft die Methode forward der Schicht auf, um die Ausgabe der Schicht im Testmodus zu berechnen
                else: # Wenn die Schicht eine andere Art von Schicht ist
                    Z = schicht.forward(Z) # Ruft die Methode forward der Schicht auf, um die Ausgabe der Schicht zu berechnen
            if i==0: # Wenn dies die erste Minibatch ist
                if self.kosten_typ=='cross-entropy': # Wenn die Kostenfunktion die Kreuzentropie ist
                    y_pred = np.argmax(Z, axis=1).tolist() # Berechnet die Vorhersagen des Modells für die Minibatch als eine Liste von Indizes
                else: # Wenn die Kostenfunktion eine andere ist
                    y_pred = Z # Speichert die Ausgabe des Modells für die Minibatch als ein Array
            else: # Wenn dies nicht die erste Minibatch ist
                if self.kosten_typ=='cross-entropy': # Wenn die Kostenfunktion die Kreuzentropie ist
                    y_pred += np.argmax(Z, axis=1).tolist() # Fügt die Vorhersagen des Modells für die Minibatch zur Liste der Vorhersagen hinzu
                else: # Wenn die Kostenfunktion eine andere ist
                    y_pred = np.vstack((y_pred, Z)) # Stapelt die Ausgabe des Modells für die Minibatch unter der bisherigen Ausgabe

        return np.array(y_pred) # Gibt die Vorhersagen des Modells für die Eingabedaten als ein Array zurück

    #Sichere alle Parameter als pkl Datei
    def save(self, fileName, saveJustParam = False):
        '''
        Speichert die Parameter des Netzes als eine pkl-Datei.

        Diese Methode liest die Parameter der einzelnen Schichten aus und speichert sie mit den Schichten in einer Liste, die mit pickle in einer Datei gespeichert wird. Wenn saveJustParam auf True gesetzt ist, werden nur die Gewichte und Bias-Werte von Dense und Conv2D-Schichten gespeichert.

        Parameters
        ----------
        fileName : str
            Der Name der Datei, in der die Parameter gespeichert werden sollen.
        saveJustParam : bool, optional
            Ob nur die Gewichte und Bias-Werte gespeichert werden sollen. Standardmäßig auf False gesetzt.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            Wenn eine Schicht einen unbekannten oder nicht unterstützten Typ hat.
        '''
        #Ließ die Parameter der einzelnen Schichten aus und speichere sie mit den Schichten in struc
        if not saveJustParam: 
            struc = [self.input_shape, ] #Struktur des Netzes mit Schichtklasse und Parameter
            for layer in self.network_structure:
                schicht = [] #Schichtklasse und Parameter
                if isinstance(layer, Conv2D): 
                    schicht.append("Conv2D")
                    param = (layer.Kernel, layer.kernel_size, layer.stride, layer.padding_type,
                    layer.activation_type, layer.use_bias, layer.weight_initializer_type,
                    layer.kernel_regularizer, layer.seed, layer.input_shape, layer.bias)
                    schicht.append(param)
                elif isinstance(layer, Pooling2D):
                    schicht.append("Pooling2D")
                    param = (layer.kernelSize, layer.stride, layer.padding_type, layer.pool_type)
                    schicht.append(param)
                elif isinstance(layer, Dense):
                    schicht.append("Dense")
                    param = (layer.neurons, layer.activation_type, layer.use_bias,
                    layer.weight_initializer_type, layer.weight_regularizer, layer.seed, layer.input_dim, layer.weight, layer.bias)
                    schicht.append(param)
                elif isinstance(layer, Flatten):
                    schicht.append("Flatten")
                    param = ()
                    schicht.append(param)
                elif isinstance(layer, Dropout):
                    schicht.append("Dropout")
                    param = (layer.p)
                    schicht.append(param)
                elif isinstance(layer, BatchNorm):
                    schicht.append("BatchNormalization")
                    param = (layer.beta, layer.eta, layer.gamma, layer.avg, layer.std, layer.alpha)
                    schicht.append(param)
                else: raise ValueError("Der folgende Schichtyp: ", type(layer), " ist unbekannt oder wird nicht unterstützt")
                struc.append(schicht)
        else:# Speichere nur die Gewichte und Bias Werte von Dens und Conv2D
            struc = []
            for layer in self.network_structure:
                schicht = []
                if isinstance(layer, Conv2D):
                    schicht.append("Conv2D")
                    param = (layer.Kernel, layer.bias)
                    schicht.append(param)
                elif isinstance(layer, Dense):
                    schicht.append("Dense")
                    param = (layer.weight, layer.bias)
                    schicht.append(param)
                else: schicht = None
                struc.append(schicht)
        print("Parameter des Netzes geladen")
         
        #Speicher die struc Liste mit pickle in fileName
        with open(fileName, "wb") as datei:
            pickle.dump(struc, datei)
        print("Parameter Abgespeichert")
    
    def load(self, fileName, loadStruc = True):
        
        """
        Lädt die Parameter des Netzes aus einer pkl-Datei.

        Diese Methode lädt eine Liste mit den Schichten und ihren Parametern aus einer Datei, die mit pickle gespeichert wurde. Wenn loadStruc auf True gesetzt ist, werden alle Schichten und ihre Parameter geladen. Wenn loadStruc auf False gesetzt ist, werden nur die Gewichte und Bias-Werte von Dense und Conv2D-Schichten geladen. Die geladenen Schichten werden dem Netz hinzugefügt.

        Parameters
        ----------
        fileName : str
            Der Name der Datei, aus der die Parameter geladen werden sollen.
        loadStruc : bool, optional
            Ob alle Schichten und ihre Parameter oder nur die Gewichte und Bias-Werte geladen werden sollen. Standardmäßig auf True gesetzt.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            Wenn eine Schicht einen unbekannten oder nicht unterstützten Typ hat.
        """
        
        #load the struc list with pickle
        with open(fileName, "rb") as datei:
            struc = pickle.load(datei)
        print("Parameter geladen")
        
        input_shape = struc[0]
        self.add(self.Input(input_shape=input_shape))
        #für jede Schicht in struc erstelle den Layer und füge ihn zum Netz hinzu
        for layer in struc[1:]:
            if layer[0] == "Conv2D":
                self.add(Conv2D(*layer[1]))
            elif layer[0] == "Pooling2D":
                self.add( Pooling2D(*layer[1]))
            elif layer[0] == "Dense":
                self.add( Dense(*layer[1]))
            elif layer[0] == "Flatten":
                self.add(Flatten())
            elif layer[0] == "Dropout":
                self.add(Dropout(layer[1]))
            elif layer[0] == "BatchNormalization":
                self.add(BatchNorm(layer[1]))      
            else: raise ValueError("Der folgende Schichtyp: ", layer[0], " ist unbekannt oder wird nicht unterstützt")
        print("Netz mit Prametern erstellt")

