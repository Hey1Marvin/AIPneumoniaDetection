'''
Diese Modul Enthält die Klassen für dei Schichten eines Neuronalen Netzes. 
definierte Schichten:
    -Dense
        Vollvermaschte/Dens Schicht
    -Conv2D
        Convolutional Layer
    -Pooling2D
        Pooling Layer (min/mean/max)
    -Dropout
        -Dropout Layer
    -Flatten
        - Flatten Layer
    -Padding
        -Padding Layer (used for Conv2D, Pooling2D)
'''


# numpy for linear algebra
import numpy as np

#Utility functions
from .Rectifier import *






# #### [Padding2D class]
#Klasse Padding2D: Fügt einen Ramen/Padding zu einem Bild hinzu
class Padding2D:
    """
    Eine Klasse, die eine Padding-Schicht für eine Faltungsschicht implementiert.

    Attributes
    ----------
    padding : {'same', 'valid'} or int or tuple of int
        Der Padding-Typ. Erlaubte Typen sind nur 'same', 'valid', eine Ganzzahl oder ein Tupel der Länge 2.
    input_shape : tuple of int
        Die Form der Eingabe zur Padding2D-Schicht
    output_shape : tuple of int
        Die Form der Ausgabe nach dem Padding
    padT : int
        Die Menge an Padding oben
    padB : int
        Die Menge an Padding unten
    padL : int
        Die Menge an Padding links
    padR : int
        Die Menge an Padding rechts

    Methods
    -------
    __init__(padding='valid')
        Der Konstruktor der Padding2D-Klasse. Initialisiert den Padding-Typ.
    get_dimensions(input_shape, kernel_size=None, stride=(1,1))
        Eine Hilfsfunktion, die die Dimension der Ausgabe nach dem Padding bestimmt.
    forward(X, kernel_size, stride=(1,1))
        Eine Funktion, die die Vorwärtspropagation durchführt. Sie berechnet die Menge an Padding, die benötigt wird, und wendet das Padding auf die Eingabe X an.
    backpropagation(dXp)
        Führt die Rückwärtspropagation durch. Sie entfernt das Padding vom Gradienten der Ausgabe.
    """

    def __init__(self, padding='valid'):
        """
        Der Konstruktor der Padding2D-Klasse. Initialisiert den Padding-Typ.

        Parameters
        ----------
        padding : {'same', 'valid'} or int or tuple of int, optional
            Der Padding-Typ. Erlaubte Typen sind nur 'same', 'valid', eine Ganzzahl oder ein Tupel der Länge 2. Standardwert ist 'valid'.
        """
        self.padding = padding

    def get_dimensions(self, input_shape, kernel_size = None, stride=(1,1)):
        """
        Eine Hilfsfunktion, die die Dimension der Ausgabe nach dem Padding bestimmt.

        Parameters
        ----------
        input_shape : tuple of int
            Die Form der Eingabe zur Padding2D-Schicht
        kernel_size : tuple of int, optional
            Die Größe des Kernels. Standardwert ist None.
        stride : tuple of int, optional
            Die Schritte entlang der Höhe und Breite (strideH, strideW). Standardwert ist (1, 1).

        Returns
        -------
        output_shape : tuple of int
            Die Form der Ausgabe nach dem Padding
        (padT, padB, padL, padR) : tuple of int
            Ein Tupel, das die Menge an Padding in alle vier Richtungen (oben, unten, links, rechts) angibt
        """
        if len(input_shape)==4:
            batch, inputC, inputH, inputW = input_shape
        elif len(input_shape)==3:
            inputC, inputH, inputW = input_shape

        
        strideH, strideW = stride
        padding = self.padding

        if type(padding)==int:
            padT, padB = padding, padding
            padL, padR = padding, padding

        if type(padding)==tuple:
            padH, padW = padding
            padT, padB = padH//2, (padH+1)//2
            padL, padR = padW//2, (padW+1)//2

        elif padding=='valid':
            padT, padB = 0, 0
            padL, padR = 0, 0

        elif padding=='same':
            # calculating how much padding is required in all 4 directions
            # (top, bottom, left and right)
            if kernel_size is None: raise ValueError("Kernel cannot be None if padding is same")
            kernelH, kernelW = kernel_size
            padH = (strideH-1)*inputH + kernelH - strideH
            padW = (strideW-1)*inputW + kernelW - strideW

            padT, padB = padH//2, (padH+1)//2
            padL, padR = padW//2, (padW+1)//2

        else:
            raise ValueError("Incorrect padding type. Allowed types are only 'same', 'valid', an integer or a tuple of length 2.")

        if len(input_shape)==4:
            output_shape = (batch, inputC, inputH+padT+padB, inputW+padL+padR)
        elif len(input_shape)==3:
            output_shape = (inputC, inputH+padT+padB, inputW+padL+padR)
        return output_shape, (padT, padB, padL, padR)

    def forward(self, X, kernel_size, stride=(1,1)):
        """
        Eine Funktion, die die Vorwärtspropagation durchführt. Sie berechnet die Menge an Padding, die benötigt wird, und wendet das Padding auf die Eingabe X an.

        Parameters
        ----------
        X : numpy.ndarray
            Die Eingabe mit Form (batch, inputC, inputH, inputW)
        kernel_size : tuple of int
            Die Kernelgröße wie in Conv2D-Schicht angegeben
        stride : tuple of int, optional
            Die Schritte entlang der Höhe und Breite (strideH, strideW). Standardwert ist (1, 1).

        Returns
        -------
        Xp : numpy.ndarray
            Das gepaddete X mit Form (batch, inputC, inputH+padT+padB, inputW+padL+padR)
        """
        self.input_shape = X.shape
        batch, inputC, inputH, inputW = self.input_shape

        self.output_shape, (self.padT, self.padB, self.padL, self.padR) = self.get_dimensions(self.input_shape,
                                                                                      kernel_size, stride=stride)

        zeros_r = np.zeros((batch, inputC, inputH, self.padR))
        zeros_l = np.zeros((batch, inputC, inputH, self.padL))
        zeros_t = np.zeros((batch, inputC, self.padT, inputW + self.padL + self.padR))
        zeros_b = np.zeros((batch, inputC, self.padB, inputW + self.padL + self.padR))

        Xp = np.concatenate((X, zeros_r), axis=3)
        Xp = np.concatenate((zeros_l, Xp), axis=3)
        Xp = np.concatenate((zeros_t, Xp), axis=2)
        Xp = np.concatenate((Xp, zeros_b), axis=2)

        return Xp

    def backpropagation(self, dXp):
        """
        Führt die Rückwärtspropagation durch. Sie entfernt das Padding vom Gradienten der Ausgabe.

        Parameters
        ----------
        dXp : numpy.ndarray
            Der Backprop-Fehler von gepaddetem X (Xp) mit Form (batch, inputC, inputH+padT+padB, inputW+padL+padR)

        Returns
        -------
        dX : numpy.ndarray
            Der Backprop-Fehler von X mit Form (batch, inputC, inputH, inputW)

        Notes
        -----
        Diese Methode verwendet die Attribute `input_shape`, `padT`, `padB`, `padL` und `padR`, die von der `forward`-Methode gesetzt wurden, um den Gradienten der Ausgabe dXp zu schneiden und das Padding zu entfernen. Sie gibt den Gradienten der Eingabe dX zurück, der die gleiche Form wie die Eingabe hat.
        """
        batch, inputC, inputH, inputW = self.input_shape
        dX = dXp[:, :, self.padT:self.padT+inputH, self.padL:self.padL+inputW]
        return dX


# #### [Convolution2D class]
class Conv2D:
    
    def __init__(self, filters, kernel_size, stride=(1, 1), padding='valid',
             activation_type=None, use_bias=True, weight_initializer_type=None,
             kernel_regularizer=None, seed=None, input_shape=None, bias = None):
            '''
            Der Konstruktor der Conv2D-Klasse. Initialisiert die Parameter der Faltungsschicht.

            Parameters
            ----------
            filters : int
                Anzahl der Ausgabefilter in der Faltung (F).
            kernel_size : int oder tuple of int
                Höhe und Breite des 2D-Faltungsfensters (kernelH, kernelW).
            stride : int oder tuple of int, optional
                Schritte entlang der Höhe und Breite (strideH, strideW). Standardwert ist (1, 1).
            padding : {'same', 'valid'} oder int oder tuple of int, optional
                Polsterungstyp. Standardwert ist 'valid'.
            activation_type : {'sigmoid', 'linear', 'tanh', 'softmax', 'prelu', 'relu'} oder None, optional
                Art der Aktivierung. Standardwert ist None.
            use_bias : bool, optional
                Gibt an, ob eine Bias-Vektor verwendet wird. Standardwert ist True.
            weight_initializer_type : str oder None, optional
                Initialisierer für die Gewichtsmatrix des Kernels. Standardwert ist None.
            kernel_regularizer : tuple of (str, float) oder None, optional
                Regularisierungsfunktion für die Kernelmatrix. Standardwert ist None.
            seed : int oder None, optional
                Für reproduzierbare Ergebnisse. Standardwert ist None.
            input_shape : tuple of int oder None, optional
                Größe des Inputs (batch, inputC, inputH, inputW). Standardwert ist None.

            Notes
            -----
            Diese Klasse implementiert eine zweidimensionale Faltungsschicht, die eine lineare Transformation auf einem lokalen Bereich der Eingabe anwendet. Die Faltung wird durch eine Gewichtsmatrix (Kernel) und einen Bias-Vektor charakterisiert, die während des Trainings gelernt werden. Die Faltung kann auch eine Aktivierungsfunktion, eine Polsterung und eine Regularisierung enthalten, um die Leistung zu verbessern.

            
            
            '''

            # Verwendung von assert-Anweisungen zur Überprüfung der Eingabetypen
            #assert isinstance(filters, int), "filters muss ein Integer sein"
            assert isinstance(kernel_size, (int, tuple)), "kernel_size muss ein Integer oder ein Tupel sein"
            assert isinstance(stride, (int, tuple)), "s muss ein Integer oder ein Tupel sein"
            assert padding in ['same', 'valid'] or isinstance(padding, (int, tuple)), "padding muss 'same', 'valid', ein Integer oder ein Tupel sein"
            assert activation_type in [None, 'sigmoid', 'linear', 'tanh', 'softmax', 'prelu', 'relu'], "Ungültiger activation_type"
            assert isinstance(use_bias, bool), "use_bias muss ein Boolean sein"
            assert kernel_regularizer is None or isinstance(kernel_regularizer, tuple), "kernel_regularizer muss ein Tupel sein"
            assert input_shape is None or isinstance(input_shape, tuple), "input_shape muss ein Tupel sein"

            self.kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
            self.stride = (stride, stride) if isinstance(stride, int) else stride
            self.padding_type = padding
            self.padding = Padding2D(padding=padding)
            self.activation_type = activation_type
            self.activation = Activation(activation_type=activation_type)
            self.use_bias = use_bias
            self.weight_initializer_type = weight_initializer_type
            self.kernel_regularizer = kernel_regularizer if kernel_regularizer is not None else ('L2', 0)
            self.seed = seed
            self.input_shape_x = input_shape
            
            #extract kernel:
            if isinstance(filters, int):
                self.F = filters
                self.Kernel = None
            elif isinstance(filters, np.ndarray):
                self.kernel_size = filters.shape[-2:]
                self.F = filters.shape[0]
                self.Kernel = filters
            else: raise ValueError("Given Filter has invalide type: ", type(filters),".")
            
            if bias is None:
                self.bias = None
            elif isinstance(bias, np.ndarray):
                self.bias = bias
            else: raise ValueError("Given Filter has invalide type: ", type(filters),".")
                

            # Extrahieren der Höhe und Breite aus kernel_size und stride
            self.kernelH, self.kernelW = self.kernel_size
            self.strideH, self.strideW = self.stride
            self.inputC, self.inputH, self.inputW = 0, 0, 0
    
    def get_dimensions(self, input_shape):
        """
        Berechnet die Dimensionen des Outputs basierend auf dem Input-Shape.

        Parameters
        ----------
        input_shape : tuple of int
            Die Form des Inputs (3D oder 4D).

        Returns
        -------
        output_shape : tuple of int
            Die Form des Outputs (3D oder 4D).

        Notes
        -----
        Diese Funktion speichert das Input-Shape und das Output-Shape als Attribute der Klasse. Sie berücksichtigt auch das Padding, das für die Faltung verwendet wird, und berechnet die Höhe und Breite des Outputs anhand des Kernels und des Schritts. Sie passt das Output-Shape an die Länge des Input-Shapes an, indem sie die Batch- und Kanalgrößen beibehält oder hinzufügt.
        """

        # Speichern des Input-Shapes
        self.input_shape_x = input_shape

        # Berechnung des Input-Shapes mit Padding, das tatsächlich für diese Conv2D verwendet wird
        self.input_shape, _ = self.padding.get_dimensions(self.input_shape_x, self.kernel_size, self.stride)

        # Entpacken des Input-Shapes in die entsprechenden Variablen
        *shape, self.inputC, self.inputH, self.inputW = self.input_shape

        # Berechnung der Output-Dimensionen
        self.outH = (self.inputH - self.kernelH) // self.strideH + 1
        self.outW = (self.inputW - self.kernelW) // self.strideW + 1

        # Zuweisung des Output-Shapes basierend auf der Länge des Input-Shapes
        self.output_shape = (*shape, self.inputC,  self.F, self.outH, self.outW) if len(input_shape) == 4 else (self.F, self.outH, self.outW)

    def initialize_parameters(self, input_shape, optimizer_type):
        """
        Diese Funktion initialisiert die Parameter der Faltungsschicht.

        Parameters
        ----------
        input_shape : tuple of int
            Form der Eingabe zur Conv2D-Schicht
        optimizer_type : str
            Art des Optimierers

        Returns
        -------
        None
            Diese Funktion gibt nichts zurück, sondern setzt die Attribute der Klasse.

        Notes
        -----
        Diese Funktion ruft die Methode `get_dimensions` auf, um die Dimensionen des Outputs basierend auf dem Input-Shape zu berechnen. Sie erstellt dann die Gewichtsmatrix (Kernel) und den Bias-Vektor mit der Klasse `Weights_initializer` und dem angegebenen Initialisierer-Typ. Sie initialisiert auch den Optimizer mit der Klasse `Optimizer` und dem angegebenen Optimizer-Typ. Sie speichert alle diese Parameter als Attribute der Klasse.
        """
        self.get_dimensions(input_shape)

        shapebias = (self.F, self.outH, self.outW)

        if self.Kernel is None:
            shape_Kernel = (self.F, self.inputC, self.kernelH, self.kernelW)

            initializer = Weights_initializer(shape=shape_Kernel, initializer_type=self.weight_initializer_type, seed=self.seed)

            self.Kernel = initializer.get_initializer()
        if self.bias is None:
            self.bias = np.zeros(shapebias)

        self.optimizer = Optimizer(optimizer_type=optimizer_type, shape_W=shape_Kernel, shape_b=shapebias)

    def dilate2D(self, X, Dr=(1,1)):
        """
        Vergrößert die Eingabe X entlang der Höhe und Breite mit einem gegebenen Dilatationsfaktor.

        Parameters
        ----------
        X : numpy.ndarray
            Eingabedaten mit Form (batch, C, H, W)
        Dr : tuple of int, optional
            Dilatationsfaktor entlang der Höhe und Breite (dh, dw). Standardwert ist (1, 1).

        Returns
        -------
        Xd : numpy.ndarray
            Dilatierte Eingabe mit Form (batch, C, H*dh, W*dw)

        Notes
        -----
        Diese Funktion verwendet die numpy.repeat-Funktion, um die Eingabe entlang der letzten beiden Achsen zu wiederholen, die der Höhe und Breite entsprechen. Sie berechnet die Anzahl der Wiederholungen anhand des Dilatationsfaktors und der Form der Eingabe. Sie gibt eine neue Ansicht auf dem Eingabearray zurück, die keinen zusätzlichen Speicherplatz benötigt.
        """
        dh, dw = Dr # Dilatationsfaktor
        batch, C, H, W = X.shape
        Xd = np.repeat(X, repeats=dw, axis=-1) # Wiederhole die Eingabe entlang der Breite
        Xd = np.repeat(Xd, repeats=dh, axis=-2) # Wiederhole die Eingabe entlang der Höhe
        return Xd # Gib die dilatierte Eingabe zurück

    def prepare_subMatrix(self, X, kernelH, kernelW, stride):
        batch, inputC, inputH, inputW = X.shape
        strideH, strideW = stride

        outH = (inputH-kernelH)//strideH + 1
        outW = (inputW-kernelW)//strideW + 1

        strides = (inputC*inputH*inputW, inputW*inputH, inputW*strideH, strideW, inputW, 1)
        strides = tuple(i * X.itemsize for i in strides)

        subM = np.lib.stride_tricks.as_strided(X,
                                               shape=(batch, inputC, outH, outW, kernelH, kernelW),
                                               strides=strides)

        return subM
        
    def convolve(self, X, kernel, stride=(1,1), mode='forward'):
        """
        Führt eine Faltung zwischen der Eingabe X und dem Kernel K aus.

        Parameters
        ----------
        X : numpy.ndarray
            Eingabedaten mit Form (batch, inputC, inputH, inputW)
        K : numpy.ndarray
            Kernelmatrix mit Form (F, inputC, kernelH, kernelW)
        s : tuple of int, optional
            Schritte entlang der Höhe und Breite (sH, sW). Standardwert ist (1, 1).
        mode : {'forward', 'backpropagation', 'backParam'} oder None, optional
            Modus der Faltung. Standardwert ist 'forward'.

        Returns
        -------
        numpy.ndarray
            Ausgabedaten mit Form (batch, F, outH, outW) im 'forward' Modus,
            (batch, inputC, inputH, inputW) im 'backpropagation' Modus oder
            (F, inputC, kernelH, kernelW) im 'backParam' Modus.

        Raises
        ------
        ValueError
            Wenn der gegebene Modus nicht erlaubt ist.

        Notes
        -----
        Diese Funktion verwendet die numpy.einsum-Funktion, um eine effiziente Faltung zwischen der Eingabe und dem Kernel durchzuführen. Sie bereitet eine Unter-Matrix der Eingabe vor, die zur Faltung verwendet wird, indem sie die Methode `prepare_subMatrix` aufruft. Sie wendet dann die Faltung an, indem sie die Einstein-Summenkonvention verwendet, die je nach Modus variiert. Sie gibt die gefaltete Ausgabe zurück, die je nach Modus eine andere Form hat.
        """
        _, _, kernelH, kernelW = kernel.shape
        subM = self.prepare_subMatrix(X, kernelH, kernelW, stride)

        match mode:
            case 'forward': return np.einsum('fckl,mcijkl->mfij', kernel, subM)
            case 'backpropagation': return np.einsum('fdkl,mcijkl->mdij', kernel, subM)
            case 'backParam': return np.einsum('mfkl,mcijkl->fcij', kernel, subM)
            case _: raise ValueError("The given mode", mode, " is not allowed, possible modes are: 'forward', 'backpropagation' and 'backParam'. ")

    
    def dZ_D_dX(self, dZ_D, imputH, inputW):
    
        # Pad the dilated dZ (dZ_D -> dZ_Dp)

        _, _, Hd, Wd = dZ_D.shape

        padH = imputH - Hd + self.kernelH - 1
        padW = inputW - Wd + self.kernelW - 1

        padding_back = Padding2D(padding=(padH, padW))

        dZ_Dp = padding_back.forward(dZ_D, self.kernel_size, self.stride)

        # Rotate K by 180 degrees

        K_rotated = self.Kernel[:, :, ::-1, ::-1]

        # convolve dZ_Dp with K_rotated

        dXp = self.convolve(dZ_Dp, K_rotated, mode='backpropagation')

        dX = self.padding.backpropagation(dXp)

        return dX

    def forward(self, X):
        # padding

        self.X = X

        Xp = self.padding.forward(X, self.kernel_size, self.stride)

        # convolve Xp with K
        Z = self.convolve(Xp, self.Kernel, self.stride, 'forward') + self.bias

        a = self.activation.forward(Z)

        return a
    
    def backpropagation(self, da):
        """
        Berechnet die Gradienten der Eingabe X, des Kernels K und des Bias-Vektors b bezüglich der Faltung mit der Aktivierungsfunktion.

        Parameters
        ----------
        da : numpy.ndarray
            Gradient der Aktivierung mit Form (batch, F, outH, outW)

        Returns
        -------
        dX : numpy.ndarray
            Gradient der Eingabe mit Form (batch, inputC, inputH, inputW)

        Notes
        -----
        Diese Funktion verwendet die `padding`-Methode, um die Eingabe X zu polstern, so dass sie mit dem Kernel und dem Schritt kompatibel ist. Sie verwendet dann die `activation`-Methode, um den Gradienten der gefalteten Ausgabe Z zu berechnen. Sie verwendet dann die `dilate2D`-Methode, um den Gradienten der Ausgabe dZ zu vergrößern, so dass er mit dem Eingang kompatibel ist. Sie verwendet dann die `dZ_D_dX`-Methode, um den Gradienten der Eingabe dX zu berechnen. Sie verwendet dann die `numpy.pad`-Funktion, um den dilatierten Gradienten dZ_D zu polstern, so dass er mit dem Kernel kompatibel ist. Sie verwendet dann die `convolve`-Methode, um die Gradienten des Kernels dK und des Bias-Vektors db zu berechnen. Sie gibt den Gradienten der Eingabe dX zurück und speichert die Gradienten des Kernels dK und des Bias-Vektors db als Attribute der Klasse.
        """
        # Polstern der Eingabe
        Xp = self.padding.forward(self.X, self.kernel_size, self.stride)
        batch, inputC, inputH, inputW = Xp.shape
        
        # Berechnung des Gradienten der gefalteten Ausgabe
        dZ = self.activation.backpropagation(da)

        # Vergrößerung des Gradienten der Ausgabe
        dZ_D = self.dilate2D(dZ, Dr=self.stride)

        # Berechnung des Gradienten der Eingabe
        dX = self.dZ_D_dX(dZ_D, inputH, inputW)

        # Berechnung der Polstergröße
        _, _, Hd, Wd = dZ_D.shape
        padH = self.inputH - Hd - self.kernelH + 1
        padW = self.inputW - Wd - self.kernelW + 1

        # Polstern des dilatierten Gradienten
        dZ_Dp = np.pad(dZ_D, pad_width=((0, 0), (0, 0), (padH // 2, padH - padH // 2), (padW // 2, padW - padW // 2)), mode='constant')

        # Berechnung des Gradienten des Kernels
        self.dK = self.convolve(Xp, dZ_Dp, mode='backParam')

        # Berechnung des Gradienten des Bias-Vektors
        self.db = np.sum(dZ, axis=0)

        # Rückgabe des Gradienten der Eingabe
        return dX
      
    def update(self, learnrate, batch, it):
        """
        Aktualisiert die Parameter der Faltungsschicht (Kernel und Bias) anhand der Gradienten und des Optimierers.

        Parameters
        ----------
        learnrate : float
            Lernrate
        batch : int
            Batch-Größe (Anzahl der Samples im Batch)
        it : int
            Iterationsnummer

        Returns
        -------
        None
            Diese Funktion gibt nichts zurück, sondern aktualisiert die Attribute der Klasse.

        Notes
        -----
        Diese Funktion verwendet die `optimizer`-Methode, um die optimierten Gradienten zu erhalten, und die `kernel_regularizer`- und `weight_regularizer`-Attribute, um die Regularisierung anzuwenden. Sie aktualisiert dann die Parameter mit der Lernrate und der Batch-Größe, indem sie die `numpy.subtract`-Funktion verwendet, die eine vektorisierte Berechnung ermöglicht.
        """
        # Erhalten der optimierten Gradienten
        dK, db = self.optimizer.get_optimization(self.dK, self.db, it)

        # Anwenden der Regularisierung
        if self.kernel_regularizer[0].casefold() == 'l2':
            dK += self.kernel_regularizer[1] * self.Kernel
        elif self.weight_regularizer[0].casefold() == 'l1':
            dK += self.kernel_regularizer[1] * np.sign(self.Kernel)

        # Aktualisieren der Parameter mit der Lernrate und der Batch-Größe
        self.Kernel = np.subtract(self.Kernel, dK * (learnrate / batch))
        if self.use_bias:
            self.bias = np.subtract(self.bias, db * (learnrate / batch))


# #### [Dropout class]

#Klasse für die Dropout Schichten/Layer
class Dropout:

    def __init__(self, p):
        '''
        Der Konstruktor der Dropout-Klasse. Initialisiert die Dropout-Wahrscheinlichkeit.
        
        Parameter:
        p: Dropout-Wahrscheinlichkeit
        
        Ausgabe:
        void
        '''
        self.p = p
        if self.p == 0:
            self.p += 1e-6
        if self.p == 1:
            self.p -= 1e-6

    def forward(self, X):
        '''
        Diese Funktion führt die Vorwärtspropagation durch. Sie erstellt eine Maske mit der gleichen Form wie X, 
        die zufällige Werte enthält, und wendet dann die Maske auf X an.
        
        Parameter:
        X: Eingabedaten
        
        Ausgabe:
        Z: Ausgabedaten nach Anwendung der Dropout-Maske
        '''
        self.mask = (np.random.rand(*X.shape) < self.p) / self.p
        Z = X * self.mask
        return Z

    def backpropagation(self, dZ):
        '''
        Diese Funktion führt die Rückwärtspropagation durch. Sie wendet die Dropout-Maske auf den Gradienten an.
        
        Parameter:
        dZ: Gradient der Ausgabedaten
        
        Ausgabe:
        dX: Gradient der Eingabedaten nach Anwendung der Dropout-Maske
        '''
        dX = dZ * self.mask
        return dX





# #### [Maxpool2D class]
class Pooling2D:
    """
    Eine Klasse, die eine Pooling-Schicht in einem künstlichen neuronalen Netzwerk implementiert.

    Attributes
    ----------
    padding : Padding2D
        Ein Objekt der Klasse Padding2D, das die Art des Padding für die Eingabedaten bestimmt.
    kernelSize : tuple of int
        Die Größe des Pooling-Kernels in Höhe und Breite. Wenn ein einzelner int-Wert angegeben wird, wird ein quadratischer Kernel angenommen.
    stride : tuple of int
        Die Schrittgröße des Pooling-Kernels in Höhe und Breite. Wenn ein einzelner int-Wert angegeben wird, wird eine gleiche Schrittgröße in beiden Richtungen angenommen.
    pool_type : str
        Die Art des Pooling, das angewendet werden soll. Mögliche Werte sind 'max', 'mean' oder 'min'.
    output_shape : tuple of int
        Die Form der Ausgabedaten nach dem Pooling.

    Methods
    -------
    get_dimensions(input_shape)
        Berechnet die Form der Ausgabedaten und das Padding für die Eingabedaten anhand der Kernel- und Schrittgröße.
    prepare_subMatrix(X, kernelSize, stride)
        Erstellt eine Matrix von Teilmatrizen, die den Pooling-Kerneln entsprechen, aus den Eingabedaten mit Hilfe von numpy-Strides.
    pooling(X, kernelSize, stride)
        Wendet das Pooling auf die Eingabedaten an und gibt die Ausgabedaten zurück.
    prepare_mask(subM, kernelH, kernelW, poolType)
        Erstellt eine Maske aus Einsen und Nullen, die angibt, welche Elemente der Teilmatrizen für das Pooling ausgewählt wurden.
    mask_dXp(mask, Xp, dZ, kernelH, kernelW)
        Multipliziert die Maske mit dem Fehlergradienten dZ, um den Fehlergradienten dXp für die gepaddeten Eingabedaten zu erhalten.
    maxmin_pool_backprop(dZ, X, type)
        Berechnet den Fehlergradienten dX für die Eingabedaten anhand des Fehlergradienten dZ und der Maske für das Max- oder Min-Pooling.
    dZ_dZp(dZ)
        Erweitert den Fehlergradienten dZ, um ihn mit dem Fehlergradienten dXp in Übereinstimmung zu bringen, indem er ihn mit Nullen auffüllt und wickelt.
    averagepool_backprop(dZ, X)
        Berechnet den Fehlergradienten dX für die Eingabedaten anhand des Fehlergradienten dZ für das Mittelwert-Pooling.
    forward(X)
        Führt das Pooling auf die Eingabedaten aus und gibt die Ausgabedaten zurück.
    backpropagation(dZ)
        Berechnet den Fehlergradienten dX für die Eingabedaten anhand des Fehlergradienten dZ und des pool_type-Attributs.
    """
    
    def __init__(self, kernelSize=(2,2), stride=(2,2), padding='valid', pool_type='mean'):
        """
        Initialisiert die Attribute der Pooling-Schicht.

        Parameters
        ----------
        kernelSize : tuple of int, optional
            Die Größe des Pooling-Kernels in Höhe und Breite. Wenn ein einzelner int-Wert angegeben wird, wird ein quadratischer Kernel angenommen. Der Standardwert ist (2,2).
        stride : tuple of int, optional
            Die Schrittgröße des Pooling-Kernels in Höhe und Breite. Wenn ein einzelner int-Wert angegeben wird, wird eine gleiche Schrittgröße in beiden Richtungen angenommen. Der Standardwert ist (2,2).
        padding : str, optional
            Die Art des Padding für die Eingabedaten. Mögliche Werte sind 'valid', 'same' oder 'full'. Der Standardwert ist 'valid'.
        pool_type : str, optional
            Die Art des Pooling, das angewendet werden soll. Mögliche Werte sind 'max', 'mean' oder 'min'. Der Standardwert ist 'mean'.

        Returns
        -------
        None
            Diese Methode gibt nichts zurück, sondern initialisiert die Attribute der Klasse.
        """
        self.padding_type = padding
        self.padding = Padding2D(padding=padding)
        
        self.kernelSize = kernelSize
        if isinstance(kernelSize, int): self.kernel_size = (kernelSize, kernelSize) 
        self.kernelH, self.kernelW = self.kernelSize

        self.stride = stride
        if isinstance(stride, int): self.stride = (stride, stride)
        self.strideH, self.strideW = self.stride

        self.pool_type = pool_type

    def get_dimensions(self, input_shape):
        """
        Berechnet die Form der Ausgabedaten und das Padding für die Eingabedaten anhand der Kernel- und Schrittgröße.

        Parameters
        ----------
        input_shape : tuple of int
            Die Form der Eingabedaten in der Form (batch, inputC, inputH, inputW), wobei batch die Batch-Größe, inputC die Anzahl der Kanäle, inputH die Höhe und inputW die Breite ist.

        Returns
        -------
        None
            Diese Methode gibt nichts zurück, sondern aktualisiert die Attribute output_shape und padding der Klasse.
        """
        #Die Größe der gepolsterten Eingabe ist die richtige input_shape
        input_shape,_ = self.padding.get_dimensions(input_shape, self.kernelSize, self.stride)
        
        *sh, inputH, inputW = input_shape

        outH = (inputH-self.kernelH)//self.strideH + 1
        outW = (inputW-self.kernelW)//self.strideW + 1
        
        self.output_shape = (*sh, outH, outW)
        
    # creates submatrixes for the pooling operation and returns them as numpy array (a view of the input matrix)
    def prepare_subMatrix(self, X, kernelSize, stride):
        """
        Erstellt eine Matrix von Teilmatrizen, die den Pooling-Kerneln entsprechen, aus den Eingabedaten mit Hilfe von numpy-Strides.

        Parameters
        ----------
        X : numpy.ndarray
            Die Eingabedaten in der Form (batch, inputC, inputH, inputW), wobei batch die Batch-Größe, inputC die Anzahl der Kanäle, inputH die Höhe und inputW die Breite ist.
        kernelSize : tuple of int
            Die Größe des Pooling-Kernels in Höhe und Breite.
        stride : tuple of int
            Die Schrittgröße des Pooling-Kernels in Höhe und Breite.

        Returns
        -------
        numpy.ndarray
            Die Matrix von Teilmatrizen in der Form (batch, inputC, Oh, Ow, kernelH, kernelW), wobei Oh die Ausgabehöhe, Ow die Ausgabebreite, kernelH die Kernelhöhe und kernelW die Kernelbreite ist.
        """
        batch, inputC, inputH, inputW = X.shape #dimension of image (batch- Batch size, inputC - channel size, inputH - height, inputW- width)
        strideH, strideW = stride #strides(Schrittweite)
        kernelH, kernelW = kernelSize #size of filter

        Oh = (inputH-kernelH)//strideH + 1    #calc height of output Matrix
        Ow = (inputW-kernelW)//strideW + 1    #calc width of output Matrix

        strides = (inputC*inputH*inputW, inputH*inputW, inputW*strideH, strideW, inputW, 1) # tuple of strides of matrix (number of lement to pass) 
        strides = tuple(i * X.itemsize for i in strides) #x.itemsize - size of the item in Byte (INT =4), list of bytes to pass for each dimension

        subM = np.lib.stride_tricks.as_strided(X, shape=(batch, inputC, Oh, Ow, kernelH, kernelW), strides=strides) #create the submatrixes (Der Codeblock subM = np.lib.stride_tricks.as_strided(X, shape=(batch, inputC, Oh, Ow, kernelH, kernelW), strides=strides) verwendet die Funktion as_strided aus dem NumPy-Modul stride_tricks, um eine neue Ansicht der Eingabematrix X zu erstellen. Diese neue Ansicht, subM, repräsentiert eine Sammlung von Unter-Matrizen, die für die Pooling-Operation in einem Convolutional Neural Network (CNN) verwendet werden können. Hier ist eine detaillierte Erklärung: X: Die ursprüngliche Eingabematrix, die die Daten für das CNN enthält. shape=(m, inputC, Oh, Ow, kernelH, kernelW): Das neue Shape-Argument definiert die Form der resultierenden Unter-Matrix subM. m ist die Anzahl der Beispiele im Batch, inputC die Anzahl der Kanäle, Oh und Ow sind die Höhe und Breite der Ausgabe nach dem Pooling, und kernelH und kernelW sind die Höhe und Breite des Pooling-Kerns. strides=strides: Das Strides-Argument gibt an, wie viele Bytes in der ursprünglichen Matrix X übersprungen werden müssen, um zum nächsten Element in jeder Dimension zu gelangen. Die Funktion as_strided ermöglicht es, ohne zusätzlichen Speicherbedarf oder Kopieren von Daten, auf die verschiedenen Regionen der Eingabematrix X zuzugreifen, als wären sie separate Unter-Matrizen. Dies ist besonders nützlich für Pooling-Operationen, bei denen ein kleiner Bereich (der Pooling-Kern) über die gesamte Eingabematrix gleitet und Operationen wie Max-Pooling oder Average-Pooling durchführt. Die resultierende Unter-Matrix subM kann dann verwendet werden, um diese Pooling-Operationen effizient durchzuführen.)
        return subM 

    #perform the pooling operation
    def pooling(self, X, kernelSize=(2,2), stride=(2,2)):
        """
        Wendet das Pooling auf die Eingabedaten an und gibt die Ausgabedaten zurück.

        Parameters
        ----------
        X : numpy.ndarray
            Die Eingabedaten in der Form (batch, inputC, inputH, inputW), wobei batch die Batch-Größe, inputC die Anzahl der Kanäle, inputH die Höhe und inputW die Breite ist.
        kernelSize : tuple of int, optional
            Die Größe des Pooling-Kernels in Höhe und Breite. Wenn ein einzelner int-Wert angegeben wird, wird ein quadratischer Kernel angenommen. Der Standardwert ist (2,2).
        stride : tuple of int, optional
            Die Schrittgröße des Pooling-Kernels in Höhe und Breite. Wenn ein einzelner int-Wert angegeben wird, wird eine gleiche Schrittgröße in beiden Richtungen angenommen. Der Standardwert ist (2,2).

        Returns
        -------
        numpy.ndarray
            Die Ausgabedaten in der Form (batch, inputC, Oh, Ow), wobei Oh die Ausgabehöhe und Ow die Ausgabebreite ist.

        Raises
        ------
        ValueError
            Wenn der pool_type-Attributwert nicht 'max', 'mean' oder 'min' ist.
        """
        subM = self.prepare_subMatrix(X, kernelSize, stride) #create view consisting of sub-matrices

        match self.pool_type:
            case 'max': return np.max(subM, axis=(-2,-1)) #Maxpooling
            case 'mean': return np.mean(subM, axis=(-2,-1)) #Average/Mean Pooling
            case 'min' : return np.min(subM, axis = (-2,-1)) #Min Pooling
            case _: raise ValueError("Allowed pool types are only 'max', 'mean' or 'min' and not", str(self.pool_type))
            
    def prepare_mask(self, subM, kernelH, kernelW, poolType = 'max'):
        """
        Erstellt eine Maske aus Einsen und Nullen, die angibt, welche Elemente der Teilmatrizen für das Max- oder Min-Pooling ausgewählt wurden.

        Parameters
        ----------
        subM : numpy.ndarray
            Die Matrix von Teilmatrizen in der Form (batch, inputC, Oh, Ow, kernelH, kernelW), wobei Oh die Ausgabehöhe, Ow die Ausgabebreite, kernelH die Kernelhöhe und kernelW die Kernelbreite ist.
        kernelH : int
            Die Höhe des Pooling-Kernels.
        kernelW : int
            Die Breite des Pooling-Kernels.
        poolType : str
            Die Art des Pooling, das angewendet werden soll. Mögliche Werte sind 'max' oder 'min'.

        Returns
        -------
        numpy.ndarray
            Die Maske in der gleichen Form wie subM.
        """
        batch, inputC, Oh, Ow, kernelH, kernelW = subM.shape # shape of submatrix (batch- Batchsize, inputC - channel size, inputH - output height, inputW- output width, kernelH - high pooling kernel, w - width pooling kernel)asd

        a = subM.reshape(-1,kernelH*kernelW)
        idx = np.where(poolType.lower() =='max', np.argmax(a, axis=1), np.argmin(a, axis=1))
        b = np.zeros(a.shape)
        b[np.arange(b.shape[0]), idx] = 1
        mask = b.reshape((batch, inputC, Oh, Ow, kernelH, kernelW))

        return mask

    def mask_dXp(self, mask, Xp, dZ, kernelH, kernelW):
        """
        Multipliziert die Maske mit dem Fehlergradienten dZ, um den Fehlergradienten dXp für die gepaddeten Eingabedaten zu erhalten.

        Parameters
        ----------
        mask : numpy.ndarray
            Die Maske in der Form (batch, inputC, Oh, Ow, kernelH, kernelW), wobei Oh die Ausgabehöhe, Ow die Ausgabebreite, kernelH die Kernelhöhe und kernelW die Kernelbreite ist.
        Xp : numpy.ndarray
            Die gepaddeten Eingabedaten in der Form (batch, inputC, inputH, inputW), wobei inputH die gepaddete Eingabehöhe und inputW die gepaddete Eingabebreite ist.
        dZ : numpy.ndarray
            Der Fehlergradient für die Ausgabedaten in der Form (batch, inputC, Oh, Ow), wobei Oh die Ausgabehöhe und Ow die Ausgabebreite ist.
        kernelH : int
            Die Höhe des Pooling-Kernels.
        kernelW : int
            Die Breite des Pooling-Kernels.

        Returns
        -------
        numpy.ndarray
            Der Fehlergradient für die gepaddeten Eingabedaten in der gleichen Form wie Xp.
        """
        dA = np.einsum('i,ijk->ijk', dZ.reshape(-1), mask.reshape(-1,kernelH,kernelW)).reshape(mask.shape)
        batch, inputC, inputH, inputW = Xp.shape
        strides = (inputC*inputH*inputW, inputH*inputW, inputW, 1)
        strides = tuple(i * Xp.itemsize for i in strides)
        dXp = np.lib.stride_tricks.as_strided(dA, Xp.shape, strides)
        #dXp = np.broadcast_to(dA, Xp.shape)
        return dXp

    #backpropagation of max pooling layer
    def maxmin_pool_backprop(self, dZ, X, type = 'max'):
        """
        Berechnet den Fehlergradienten dX für die Eingabedaten anhand des Fehlergradienten dZ und der Maske für das Max- oder Min-Pooling.

        Parameters
        ----------
        dZ : numpy.ndarray
            Der Fehlergradient für die Ausgabedaten in der Form (batch, inputC, Oh, Ow), wobei Oh die Ausgabehöhe und Ow die Ausgabebreite ist.
        X : numpy.ndarray
            Die Eingabedaten in der Form (batch, inputC, inputH, inputW), wobei inputH die Eingabehöhe und inputW die Eingabebreite ist.
        type : str
            Die Art des Pooling, das angewendet werden soll. Mögliche Werte sind 'max' oder 'min'.

        Returns
        -------
        numpy.ndarray
            Der Fehlergradient für die Eingabedaten in der Form (batch, inputC, inputH, inputW).
        """
        Xp = self.padding.forward(X, self.kernelSize, self.stride)

        subM = self.prepare_subMatrix(Xp, self.kernelSize, self.stride)

        _, _, _, _, kernelH, kernelW = subM.shape

        mask = self.prepare_mask(subM, kernelH, kernelW, type)

        dXp = self.mask_dXp(mask, Xp, dZ, kernelH, kernelW)

        return dXp
    
    
    def dZ_dZp2(self, dZ):
        """
        Erweitert den Fehlergradienten dZ, um ihn mit dem Fehlergradienten dXp in Übereinstimmung zu bringen, indem er ihn mit Nullen auffüllt und wickelt.

        Parameters
        ----------
        dZ : numpy.ndarray
            Der Fehlergradient für die Ausgabedaten in der Form (batch, inputC, Oh, Ow), wobei Oh die Ausgabehöhe und Ow die Ausgabebreite ist.

        Returns
        -------
        numpy.ndarray
            Der erweiterte Fehlergradient in der Form (batch, inputC, inputH, inputW), wobei inputH die gepaddete Eingabehöhe und inputW die gepaddete Eingabebreite ist.
        """
        strideH, strideW = self.stride
        kernelH, kernelW = self.kernelSize
        dZp = np.pad(dZ, pad_width=((0, 0), (0, 0), (0, kernelH-1), (0, kernelW-1)), mode='constant')
        dZp = np.pad(dZp, pad_width=((0, 0), (0, 0), (0, -strideH+1), (0, -strideW+1)), mode='wrap')
        return dZp
    
    def dZ_dZp(self, dZ):
        strideh, stridew = self.stride
        kernelH, kernelW = self.kernelSize

        dZp = np.kron(dZ, np.ones((kernelH,kernelW), dtype=dZ.dtype)) # similar to repelem in matlab

        jh, jw = kernelH-strideh, kernelW-stridew # jump along height and width

        if jw!=0:
            L = dZp.shape[-1]-1

            l1 = np.arange(stridew, L)
            l2 = np.arange(stridew + jw, L + jw)

            mask = np.tile([True]*jw + [False]*jw, len(l1)//jw).astype(bool)

            r1 = l1[mask[:len(l1)]]
            r2 = l2[mask[:len(l2)]]

            dZp[:, :, :, r1] += dZp[:, :, :, r2]
            dZp = np.delete(dZp, r2, axis=-1)

        if jh!=0:
            L = dZp.shape[-2]-1

            l1 = np.arange(strideh, L)
            l2 = np.arange(strideh + jh, L + jh)

            mask = np.tile([True]*jh + [False]*jh, len(l1)//jh).astype(bool)

            r1 = l1[mask[:len(l1)]]
            r2 = l2[mask[:len(l2)]]

            dZp[:, :, r1, :] += dZp[:, :, r2, :]
            dZp = np.delete(dZp, r2, axis=-2)
        
        return dZp
    
    
    #backpropagation of the avarage pooling layer
    def averagepool_backprop2(self, dZ, X):
        """
        Berechnet den Fehlergradienten dX für die Eingabedaten anhand des Fehlergradienten dZ für das Mittelwert-Pooling.

        Parameters
        ----------
        dZ : numpy.ndarray
            Der Fehlergradient für die Ausgabedaten in der Form (batch, inputC, Oh, Ow), wobei Oh die Ausgabehöhe und Ow die Ausgabebreite ist.
        X : numpy.ndarray
            Die Eingabedaten in der Form (batch, inputC, inputH, inputW), wobei inputH die Eingabehöhe und inputW die Eingabebreite ist.

        Returns
        -------
        numpy.ndarray
            Der Fehlergradient für die Eingabedaten in der Form (batch, inputC, inputH, inputW).
        """
        Xp = self.padding.forward(X, self.kernelSize, self.stride)

        batch, inputC, inputH, inputW = Xp.shape
        dZp = self.dZ_dZp(dZ)
        
        padH = inputH - dZp.shape[-2]
        padW = inputW - dZp.shape[-1]

        padding_back = Padding2D(padding=(padH, padW))

        dXp = padding_back.forward(dZp, stride=self.stride, kernel_size=self.kernelSize)

        #return dXp /(inputH*inputW) # (self.strideH*self.strideW)
        return np.mean(dXp, axis=(-2, -1), keepdims=True)
    
    def averagepool_backprop(self, dZ, X):
        batch, inputC, inputH, inputW = X.shape
        kernelH, kernelW = self.kernelSize
        strideH, strideW = self.stride

        dX = np.zeros_like(self.padding.forward(X, self.kernelSize, self.stride))

        for h in range(dZ.shape[2]):
            for w in range(dZ.shape[3]):
                h_start = h * strideH
                h_end = h_start + kernelH
                w_start = w * strideW
                w_end = w_start + kernelW

                dX[:, :, h_start:h_end, w_start:w_end] += dZ[:, :, h, w][:, :, np.newaxis, np.newaxis]

        dX /= (kernelH * kernelW)

        return dX
    
    
    def forward(self, X):
        """
        Führt das Pooling auf die Eingabedaten aus und gibt die Ausgabedaten zurück.

        Parameters
        ----------
        X : numpy.ndarray
            Die Eingabedaten in der Form (batch, inputC, inputH, inputW), wobei batch die Batch-Größe, inputC die Anzahl der Kanäle, inputH die Höhe und inputW die Breite ist.

        Returns
        -------
        numpy.ndarray
            Die Ausgabedaten in der Form (batch, inputC, Oh, Ow), wobei Oh die Ausgabehöhe und Ow die Ausgabebreite ist.
        """

        self.X = X
                        
        # padding
        Xp = self.padding.forward(X, self.kernelSize, self.stride)

        Z = self.pooling(Xp, self.kernelSize, self.stride)
        return Z

    def backpropagation(self, dZ):
        """
        Berechnet den Fehlergradienten dX für die Eingabedaten anhand des Fehlergradienten dZ und des pool_type-Attributs.

        Parameters
        ----------
        dZ : numpy.ndarray
            Der Fehlergradient für die Ausgabedaten in der Form (batch, inputC, Oh, Ow), wobei Oh die Ausgabehöhe und Ow die Ausgabebreite ist.

        Returns
        -------
        numpy.ndarray
            Der Fehlergradient für die Eingabedaten in der Form (batch, inputC, inputH, inputW).
        """
        if self.pool_type=='max':
            dXp = self.maxmin_pool_backprop(dZ, self.X)
        elif self.pool_type=='mean':
            dXp = self.averagepool_backprop(dZ.copy(), self.X)
            #dXp2 = self.maxmin_pool_backprop(dZ, self.X)
            #print("X-shape: ", self.X.shape, "  dXp-shape: ", dXp.shape, "  dxp2-shape: ", dXp2.shape)
            #dXp2 = self.averagepool_backprop2(dZ.copy(), self.X) #optimized
        dX = self.padding.backpropagation(dXp)
        return dX


# #### Flatten class
#4D zu 2D


class Flatten:
    """
    Eine Klasse, die eine Flattening-Schicht in einem künstlichen neuronalen Netzwerk implementiert.

    Attributes
    ----------
    batch : int
        Die Batch-Größe der Eingabedaten.
    inputC : int
        Die Anzahl der Kanäle der Eingabedaten.
    inputH : int
        Die Höhe der Eingabedaten.
    inputW : int
        Die Breite der Eingabedaten.
    output_shape : int
        Die Form der Ausgabedaten nach dem Flattening.

    Methods
    -------
    forward(X)
        Flacht die Eingabedaten ab und gibt die Ausgabedaten zurück.
    backpropagation(dZ)
        Formt den Fehlergradienten dZ zurück in die Form der Eingabedaten und gibt den Fehlergradienten dX zurück.
    get_dimensions(input_shape)
        Bestimmt die Form der Eingabe- und Ausgabedaten anhand der input_shape.
    """

    def __init__(self):
        """
        Initialisiert die Attribute der Flattening-Schicht.

        Parameters
        ----------
        None
            Diese Methode nimmt keine Parameter entgegen.

        Returns
        -------
        None
            Diese Methode gibt nichts zurück, sondern initialisiert die Attribute der Klasse.
        """
        pass

    def forward(self, X):
        """
        Flacht die Eingabedaten ab und gibt die Ausgabedaten zurück.

        Parameters
        ----------
        X : numpy.ndarray
            Die Eingabedaten in der Form (batch, inputC, inputH, inputW), wobei batch die Batch-Größe, inputC die Anzahl der Kanäle, inputH die Höhe und inputW die Breite ist.

        Returns
        -------
        numpy.ndarray
            Die Ausgabedaten in der Form (batch, inputC * inputH * inputW).
        """
        self.batch, self.inputC, self.inputH, self.inputW = X.shape
        X_flat = X.reshape((self.batch, self.inputC * self.inputH * self.inputW))
        return X_flat

    def backpropagation(self, dZ):
        """
        Formt den Fehlergradienten dZ zurück in die Form der Eingabedaten und gibt den Fehlergradienten dX zurück.

        Parameters
        ----------
        dZ : numpy.ndarray
            Der Fehlergradient für die Ausgabedaten in der Form (batch, inputC * inputH * inputW).

        Returns
        -------
        numpy.ndarray
            Der Fehlergradient für die Eingabedaten in der Form (batch, inputC, inputH, inputW).
        """
        dX = dZ.reshape((self.batch, self.inputC, self.inputH, self.inputW))
        return dX

    def get_dimensions(self, input_shape):
        """
        Bestimmt die Form der Eingabe- und Ausgabedaten anhand der input_shape.

        Parameters
        ----------
        input_shape : tuple of int
            Die Form der Eingabedaten in der Form (batch, inputC, inputH, inputW) oder (inputC, inputH, inputW), wobei batch die Batch-Größe, inputC die Anzahl der Kanäle, inputH die Höhe und inputW die Breite ist.

        Returns
        -------
        None
            Diese Methode gibt nichts zurück, sondern aktualisiert die Attribute batch, inputC, inputH, inputW und output_shape der Klasse.
        """
        if len(input_shape)==4:
            self.batch, self.inputC, self.inputH, self.inputW = input_shape
        elif len(input_shape)==3:
            self.inputC, self.inputH, self.inputW = input_shape

        self.output_shape = self.inputC * self.inputH * self.inputW
        


# Klasse für die vollvernetzten/Dens - Layer
class Dense:

    def __init__(self, neurons, activation_type=None, use_bias=True,
                 weight_initializer_type=None, weight_regularizer=None, seed=None, input_dim=None, weights = None, bias = None):

        '''
        Parameters:

        neurons: (positiver int) Anzahl der Neuronen

        activation_type: Art der Aktivierung, Standard: linear,  mögliche Optionen : 'sigmoid', 'linear', 'tanh', 'softmax', 'prelu', 'relu' ...

        use_bias: (boolean) gibt an, ob der Bias genutz werden soll

        weight_initializer_type: (str) Initzialisierer für die Gewichte

        weight_regularizer: (tuple) Regularisierer, welcher auf die Matrix angewendet wird möglic: ('L2', 0.01) or ('L1', 2)

        seed: (seed) um das gleiche Netz zu repruduzieren

        input_dim: (int) Anzahl der Neuronen der Eingabeschicht
        '''
        self.neurons = neurons
        self.activation_type = activation_type
        self.activation = Activation(activation_type=activation_type) 
        self.use_bias = use_bias 
        self.weight_initializer_type = weight_initializer_type 
        if weight_regularizer is None:
            self.weight_regularizer = ('L2', 0)
        else:
            self.weight_regularizer = weight_regularizer
        self.seed = seed
        self.input_dim = input_dim
        
        if weights is None:
            self.weight = None
        elif isinstance(weights, np.ndarray):
            self.weight = weights
            self.neurons = weights.shape[-1]
            self.input_shape = weights.shape[0]
        else: raise ValueError("The given weight type is invalide: ", type(weights))

        if bias is None:
            self.bias = None
        elif isinstance(bias, np.ndarray):
            self.bias = bias
        else: raise ValueError("The given weight type is invalide: ", type(bias))
        
        if isinstance(self.weight, np.ndarray) and isinstance(self.bias, np.ndarray) and self.bias.shape != (self.neurons, 1): raise ValueError("The bias Array has the wrong shape of: ", self.bias.shape)

    def initialize_parameters(self, input_shape, optimizer_type):
        '''
        Diese Funktion initialisiert die Gewichte und den Bias des Neurons und den Optimierer.
        
        Parameter:
        input_shape: (tuple) gibt die Eingabegröße bzw. Neuronen des vorigen Layers an
        optimizer_type: (str) gibt den Typen des Optimizer an
        
        Ausgabe:
        Keine direkte Ausgabe, aber sie aktualisiert die Attribute self.W, self.b und self.optimizer der Klasse.
        '''
        shapeWeight = (input_shape, self.neurons) #Größe der Gewichte
        shapebias = (self.neurons, 1) #Größe (shape) der Bias-Werte
        if self.weight is None:
            initializer = Weights_initializer(shape=shapeWeight, initializer_type=self.weight_initializer_type, seed=self.seed) #Initialisierer für die Gewichte
            self.weight = initializer.get_initializer() #Initialiseren der Gewichte
        if self.bias is None:
            self.bias = np.zeros(shapebias) #Initialisieren der Bias-Werte

        self.optimizer = Optimizer(optimizer_type=optimizer_type, shape_W=shapeWeight, shape_b=shapebias)

    def forward(self, X):
        '''
        Diese Funktion führt die Vorwärtspropagation durch.
        
        Parameter:
        X: Eingabedaten
        
        Ausgabe:
        out: Aktivierungswerte nach Anwendung der Aktivierungsfunktion
        '''
        #out = f(x_i * wi + b)
        self.X = X
        d = np.dot(X, self.weight) # Multiplizieren der Eingaben mit den Gewichten jedes Neurons
        self.s = d + self.bias.T #addiern des Bias-Wertes
        out = self.activation.forward(self.s) #Akktivierungsfunktion anwenden
        return out

    def backpropagation(self, error):
        '''
        Diese Funktion führt die Rückwärtspropagation durch.
        
        Parameter:
        da: Fehler-Gradient der nächsten Schicht
        
        Ausgabe:
        dX: Fehldergradient nach der Eingabe, welche für die vorigen Schichten benötigt wird
        '''
        #dX = f'(error_i) * w_i
        dz = self.activation.backpropagation(error) #Ableitung der Aktivierungsfunktion von dem Fehler der nächsten Schicht
        dr = dz.copy()
        self.dbias = np.sum(dz, axis=0).reshape(-1,1) #Änderung der Gewichte
        self.dweight = np.dot((self.X.T), dr) #Änderung der Bias-Werte
        dX = np.dot(dr, (self.weight.T))
        return dX

    def update(self, learnrate, batch, k):
        '''
        Diese Funktion aktualisiert die Gewichte und den Bias basierend auf den während der Rückwärtspropagation berechneten Ableitungen.
        
        Parameter:
        learnrate: Lernrate
        batch: Batch-Größe (Anzahl der Proben im Batch)
        k: Iterationsnummer
        
        Ausgabe:
        void
        '''

        dW, db = self.optimizer.get_optimization(self.dweight, self.dbias, k)
        #Anwenden des Geweichtsregularisierers
        if self.weight_regularizer[0].lower()=='l2':
            dW += self.weight_regularizer[1] * self.weight
        elif self.weight_regularizer[0].lower()=='l1':
            dW += self.weight_regularizer[1] * np.sign(self.weight)

        self.weight -= dW*(learnrate/batch)
        if self.use_bias:
            self.bias -= db*(learnrate/batch)


