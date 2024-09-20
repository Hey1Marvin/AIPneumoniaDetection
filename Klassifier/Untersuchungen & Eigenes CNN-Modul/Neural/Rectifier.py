#used Modules:
import numpy as np



#content of this Modul:
# #### [Activation class]

# #### [Cost function]

# #### [Optimizers]

# #### [Learning Rate decay]

# #### [Utility function]

# #### [Weights initializer class]







class Activation:
    def __init__(self, activation_type= "linear"):
        self.activation_type = activation_type
        self.funcObj = recPicker(activation_type)
        self.forw = self.funcObj.func
        self.backw = self.funcObj.deriv
    
    def forward(self, X):
        self.X = X
        z = self.forw(X)
        return z
    
    def backpropagation(self, dz):
        f_prime = self.backw(self.X)
        if self.activation_type=='softmax':
            # because derivative of softmax is a tensor
            dx = np.einsum('ijk,ik->ij', f_prime, dz)
        else:
            dx = dz * f_prime
        return dx


'''
Parameters

x: input matrix of shape (m, d)
where 'm' is the number of samples (in case of batch gradient descent of size m)
and 'd' is the number of features
'''
class Function():
    def __init__(self):
        pass
    
    def func(self, x):
        return x
    
    def deriv(self,x):
        return 1

#replaces the derive function with f(x) = 1
class NoDeriv(Function): 
    def __init__(self, f='sigmoid'):
        self.f = recPicker(float)
        
    def func(self, x):
        return self.f.func(x)
    
    def deriv(self, x): 
        return 1
    
    def change(self,func):
        self.f = func
        return self

#Sigmoid Funktionen
class Sigmoid(Function):
    def func(self, x):
        return 1 / (1 + np.exp(-x))
    
        
    def deriv(self, x):
        return self.func(x) * (1 - self.func(x))
    
class BSigmoid(Function): #Bipolares Sigmoid f(x) = (1 - e^(-x)) / (1 + e^(-x))
    def func(self,x):
        return (1 - np.exp(-x)) / (1 + np.exp(-x))
    
    def deriv(self,x):
        return 1 / (2 * (0.5 * (np.exp(0.5*x) + np.exp(-0.5*x)))**2)


#Tanh Funktionen
class Tanh(Function):
    def func(self, x):
        z = np.exp(x)
        return (z - 1/z) / (z + 1/z)
    
    def deriv(self, x):
        return 1 - (self.func(x))**2
class LCTanh(Function): #LeCuns Tanh f(x) = 1.7159 * tanh((2/3) * x)
    def __init__(self):
        self.tanh = Tanh()
        
    def func(self,x):
        return 1.7159 * self.tanh.func((2/3)*x)
    
    def deriv(self,x):
        return 1.7159 * self.tanh.deriv((2/3)*x) * (2/3)   
class HTanh(Function): #Hard Tanh f(x) = max(-1, min(1,x))
    def __init__(self,start=-1, end= 1):
        self.s = start
        self.e = end
    
    def func(self, x):
        min = np.where(x>self.e, self.e)
        return np.where(self.s< min, self.s)
    
    def intfunc(self, x):
        return max(self.s, min(x, self.e))
    
    def deriv(self, x):
        #kx = np.where(-1<x<1, 1)
        #return np.where(kx<=-1 or kx>=1, 0)
        return x * (-1 < x < 1)
        
    def intderiv(self,x):
        if -1 < x < 1:return 1
        else: return 0
           
    

#Relu Funktionen    
class ReLU(Function):
    def func(self,x): 
        return x * (x>0)
    
    def deriv(self, x):
        return (x>0) * np.ones(x.shape)
        
    def intderiv(self, x):
        if x > 0: return 0
        return 1
    
class LReLU(Function): #Leaky ReLU
    def __init__(self, a = 0.2):
        self.a = a #slope Parameter
    
    def func(self, x, alpha = None):
        if alpha is None: alpha = self.a
        return np.where(x>0, x, alpha*x)
        
    def deriv(self, x, alpha):
        if alpha is None: alpha = self.a
        return np.where(x>0, 1, alpha)
    
    def intfunc(self, x, alpha = None):
        if alpha is None: alpha = self.a
        if x > 0: return x
        return alpha*x
    
    def intderiv(self, x, alpha = None):
        if alpha is None: alpha = self.a
        if x > 0: return 1
        return alpha
    
class CosReLU(Function): #1. Modifikation mit cos von Relu gut für mnist f(x) = max(0,x) + cos(x)
    def __init__(self,wende=0):
        self.w = wende #Wendepunkt
    
    def func(self, x):
        return x*(x>0) + np.cos(x)

    def deriv(self, x):
        return np.where(x>0, 1-np.sin(x), -np.sin(x))
    
    def intderiv(self,x):
        if x>0: return   1 - np.sin(x)
        return -np.sin(x)
class SinReLU(Function): #2.Modifikation mit sin
    def __init__(self,wende=0):
        self.w = wende #Wendepunkt
    
    def func(self, x):
        return x*(x>0) + np.cos(x)
    def deriv(self, x):
        return np.where(x>0, 1-np.cos(x), -np.cos(x))
    
    def intderiv(self,x):
        if x>0: return  1 + np.cos(x)
        return np.cos(x)
    
class SReLU(Function): #Smooth rectified Linear Unit/Smooth Max/Soft Plus
    def func(self, x):
        return np.log(1 + np.exp(x))

    def deriv(self,x):
        return 1/(1 + np.exp(-x))


#Lineare Funktionen
class Linear(Function):
    def func(self, x):
        return x
    def deriv(self, x):
        return np.ones(x.shape)
    
class Linear2(Function):
    def __init__(self, m=1, n=0):
        self.m = m
        self.n = n
    
    def func(self, x):
        return self.m * x + self.n
    def deriv(self, x):
        return np.ones(x.shape)
    
    def intderiv(self, x):
        return self.m
    
class PwLinear(Function): #piecewise(Stückweise) Linear
    def __init__(self, start, end, before = 0, after = 1):
        self.s = start #Start Stelle der Liniaren Funktion
        self.e = end #Ende der Liniaren Funktion
        self.b = before # f(x) wenn x< start
        self.a = after #fx) wenn x > end
        self.m = (after - before) / (end - start) #Anstieg der Linearen Funktion von start to end m = (y2 - y1)/(x2 - x1)
        self.n = before - (self.m * start) #Schnittstelle mit de y-Achse n = y - (m * x)
    
    def update(self):
        self.m = (self.a - self.b) / (self.e - self.s) #Anstieg der Linearen Funktion von start to end m = (y2 - y1)/(x2 - x1)
        self.n = self.b - (self.m * self.s) #Schnittstelle mit de y-Achse n = y - (m * x)
    
    def func(self, x):
        kx = np.where(self.start<x<self.e, x*self.m+self.n)
        kkx = np.where(self.e<kx, self.a)
        return np.where(self.s>kkx, self.b)
    def deriv(self, x):
        return np.where(self.s<x<self.e, self.m, 0)
    
    def intfunc(self, x):
        if x < self.s: return self.b
        elif x > self.e: return self.a
        else: return x*self.m + self.n
    
    def intderiv(self, x):
        if x < self.s: return 0
        elif x > self.e: return 0
        else: return self.m


class Step(Function):
    def __init__(self, step = 0, b = 0, a=1): #f(x) = { b if x<=step; a if x>step}
        self.step = step
        self.b = b
        self.a = a
    
    def func(self, x, step = None, a = None, b = None):
        if step is None: step = self.step
        if a is None: a = self.a
        if b is None: b = self.b
        return np.where(x> np.step, self.a, self.b)
    
    def deriv(self, x, step = None, a = None, b= None):
        if step is None: step = self.step
        if a is None: a = self.a
        if b is None: b = self.b
        return np.zeros(x.shape)
    
    def intfunc(self,x):
        if x > self.step: return self.a
        return self.b
    
    def deriv(self,x):
        return 0
        
class Abs(Function):
    def func(self, x):
        if x<0: return -x
        return np.abs(x)
    
    def deriv(self, x):
        return np.ones(x.shape)
    
    def intderiv(self, x):
        return 1

#Andere Funktionen
class Softmax(Function):
    def func(self,x):
        '''
        Parameters

        x: input matrix of shape (m, d)
        where 'm' is the number of samples (in case of batch gradient descent of size m)
        and 'd' is the number of features
        '''
        z = x - np.max(x, axis=-1, keepdims=True)
        numerator = np.exp(z)
        denominator = np.sum(numerator, axis=-1, keepdims=True)
        softmax = numerator / denominator
        return softmax
    
    def softmax_grad(s): 
        # Take the derivative of softmax element w.r.t the each logit which is usually Wi * X
        # input s is softmax value of the original input x. 
        # s.shape = (1, n) 
        # i.e. s = np.array([0.3, 0.7]), x = np.array([0, 1])

        # initialize the 2-D jacobian matrix.
        jacobian_m = np.diag(s)

        for i in range(len(jacobian_m)):
            for j in range(len(jacobian_m)):
                if i == j:
                    jacobian_m[i][j] = s[i] * (1-s[i])
                else: 
                    jacobian_m[i][j] = -s[i]*s[j]
        return jacobian_m
    
    def deriv(self, x):
        '''
        Parameters

        x: input matrix of shape (m, d)
        where 'm' is the number of samples (in case of batch gradient descent of size m)
        and 'd' is the number of features
        '''
        if len(x.shape)==1:
            x = np.array(x).reshape(1,-1)
        else:
            x = np.array(x)
        m, d = x.shape
        a = self.func(x)
        tensor1 = np.einsum('ij,ik->ijk', a, a)
        tensor2 = np.einsum('ij,jk->ijk', a, np.eye(d, d))
        return tensor2 - tensor1

class KLL(Function): #komplementäres Log-Log f(x) = 1 - e^(-e^x)
    def func(self,x):
        return  1 - np.exp(-np.exp(x))
    
    def deriv(self,x):
        return np.exp(-np.exp(x))

class Logit(Function):
    def func(self, x):
        return np.log((x/(1-x)))
    
    def deriv(self, x):
        return (-1) / (x*(x-1))
    
#class Probit(Function):

class Cosinus(Function):
    def func(self,x):
        return np.cos(x)
    def deriv(self,x): 
        return -np.sin(x)

class Sinus(Function):
    def func(self,x):
        return np.sin(x)
    def deriv(self,x): 
        return np.cos(x)



#Radiale Basisfunktion Netzwerkaktivierungsfunktionen
class Gaussche(Function):
    def func(self, x):
        return np.exp(-0.5 * (x**2))
    
    def deriv(self,x):
        return -x * np.exp(-0.5 * x**2)

class Normal(Function): #Normalverteilung / Dichtefunktion
    def __init__(self, mhy, phi):
        self.m = mhy
        self.p = phi
    
    def func(self,x):
        return (1/(np.sqrt(2*np.pi * (self.p**2))))  * np.exp(-0.5 * ((x-self.m)/self.p)**2)
    
    def deriv(self, x):
        return ((self.m * np.exp((-self.m**2)/(2* self.p**2)) * np.sqrt(2)) / (2 * np.abs(self.p**3) * np.sqrt(np.pi))    -    (x * np.exp((-self.m**2)/(2* self.p**2)) * np.sqrt(2)) / (2 * np.abs(self.p**3) * np.sqrt(np.pi)))    *    np.exp(((self.m * x)/(self.p**2))  -  ((x**2) / (2 * self.p**2)))

class Multiquatratisch(Function):#Abstan (x,0) zu Punkt (x,y)
    def __init__(self,dot):
        self.x = dot[0]
        self.y = dot[1]
    
    def func(self, x):
        return np.sqrt((x - self.x)**2 + self.y**2)
    
    def deriv(self, x):
        return (x - self.x)  /  (np.sqrt(x**2 - 2*x*self.x + self.y**2 + self.x**2))
       
       
class IMultiquatratisch(Function):#Inverse multiquatratisch
    def __init__(self,dot):
        self.x = dot[0]
        self.y = dot[1]
    
    def func(self, x):
        return 1 / np.sqrt((x - self.x)**2 + self.y**2)
    
    def deriv(self, x):
        return -(x - self.x)  /  (np.sqrt(x**2 - 2*x*self.x + self.y**2 + self.x**2) ** (3/2))
    


def recPicker(name):
    #Bezeichnungen welche für die Funktionen eingegeben werden koennen
    names = [["noderive","replacederive","noderiv"],["sigmoid","sig"], ["bsigmoid", "bipolaressigmoid"], ["tanh","tangenshyperbolicus"], ["lctanh", "lecunstanh", "lecunstangenshyperbolicus", "lctangenshyperbolicus", "lecun'stangenshyperbolicus", "lecun'stanh"], ["htanh", "hardtanh", "htangenshyperbolicus", "hardtangenshyperbolicus"], ["relu", "rectifiedlinearunit", "gleichrichter", "maxfunktion", "rampenfunktion"], ["lrelu", "lrectifiedlinearunit", "lgleichrichter", "leakyrelu", "leakyrectifiedlinearunit", "leakygleichrichter", "lmaxfunktion", "lrampenfunktion", "leakymaxfunktion", "leakyrampenfunktion"], ["mcrelu", "mcrectifiedlinearunit", "mcgleichrichter", "mcmaxfunktion", "mcrampenfunktion", "crelu", "crectifiedlinearunit", "cgleichrichter", "cmaxfunktion", "crampenfunktion", "cosrelu", "cosrectifiedlinearunit", "cosgleichrichter", "cosmaxfunktion", "cosrampenfunktion", "cosinusrelu", "cosinusrectifiedlinearunit", "cosinusgleichrichter", "cosinusmaxfunktion", "cosinusrampenfunktion"], ["sinrelu", "sinrectifiedlinearunit", "singleichrichter", "sinmaxfunktion", "sinrampenfunktion", "sinusrelu", "sinusrectifiedlinearunit", "sinusgleichrichter", "sinusmaxfunktion", "sinusrampenfunktion"], ["srelu", "srectifiedlinearunit", "sgleichrichter", "smaxfunktion", "srampenfunktion", "smoothrelu", "smoothrectifiedlinearunit", "smoothgleichrichter", "smoothmaxfunktion", "smoothrampenfunktion"], ["linear", "lin"], ["pwlinear","piecewiselinear", "pwlin"], ["step", "schritt"], ["abs","absolute", "absolut"],["softmax","normalisiertesexpotential", "normalizeexpotential"],["kll", "komplementäresloglog", "komplementäreslog-log","klog-log","kloglog"], ["logit"], ["c","cos","cosinus"], ["s","sin", "sinus"], ["gaussche","gauss"], ["normal", "dichtefunktion","normalfunktion"], ["multiquatratisch"], ["imultiquatratisch", "inversemultiquatratisch"]]
    #Bezeichungen der Funktionen in diesem Script
    functions = [NoDeriv, Sigmoid, BSigmoid, Tanh, LCTanh, HTanh, ReLU, LReLU, CosReLU, SinReLU, SReLU, Linear, PwLinear, Step, Abs, Softmax, KLL, Logit, Cosinus, Sinus, Gaussche, Normal, Multiquatratisch, IMultiquatratisch]
    
    #in der Liste nach passender Funktion suchen und diese zurueckgeben
    for i in range(len(names)):
        if name.lower() in names[i]:
            return functions[i]()
    return False







class Cost:
    
    names = ['mse', 'cross-entropy', 'weighted', 'log', 'kullback-leibler', 'exp', 'hinge', 'huber'] #possible cost functions to choose from

    def __init__(self, cost_type='mse'):
        '''
        Parameters

        cost_type: type of cost function
        available options are 'mse', and 'cross-entropy'
        '''
        self.cost_type = cost_type

    def mse(self, a, y):
        '''
        Parameters

        a: Predicted output array of shape (batch, d)
        y: Actual output array of shape (batch, d)
        '''
        return (1/2)*np.sum((np.linalg.norm(a-y, axis=1))**2)

    def d_mse(self, a, y):
        '''
        represents dJ/da

        Parameters

        a: Predicted output array of shape (batch, d)
        y: Actual output array of shape (batch, d)
        '''
        return a - y

    def cross_entropy(self, a, y, epsilon=1e-12):
        '''
        Parameters

        a: Predicted output array of shape (batch, d)
        y: Actual output array of shape (batch, d)
        '''
        a = np.clip(a, epsilon, 1. - epsilon)
        return -np.sum(y*np.log(a))

    def d_cross_entropy(self, a, y, epsilon=1e-12):
        '''
        represents dJ/da

        Parameters

        a: Predicted output array of shape (batch, d)
        y: Actual output array of shape (batch, d)
        '''
        a = np.clip(a, epsilon, 1. - epsilon)
        return -y/a

    def binary_crossentropy(self, y_out, y, *args):
        '''
        Berechnet die Binary Cross-Entropy-Fehlerfunktion.

        Parameter:
        y: Tatsächliche Ausgabe-Array der Form (batch, d)
        y_out: Vorhergesagter Ausgabe-Array der Form (batch, d)

        Rückgabe:
        BCE-Fehlerwert
        '''
        return -1 * (y * np.log(y_out) + (1 - y) * np.log(1 - y_out))

    def d_binary_crossentropy(self, y_out, y, *args):
        '''
        Berechnet die Ableitung der Binary Cross-Entropy-Fehlerfunktion.

        Parameter:
        y: Tatsächliche Ausgabe-Array der Form (batch, d)
        y_out: Vorhergesagter Ausgabe-Array der Form (batch, d)

        Rückgabe:
        Ableitung der BCE-Fehlerfunktion
        '''
        return y_out - y / (y_out * (1 - y_out))
    
    def weighted(self, a, y, c):
        '''
        Parameters

        a: Predicted output array of shape (batch, d)
        y: Actual output array of shape (batch, d)
        c: Weight factor array of shape (batch, 1)
        '''
        return c * (a - y)**2

    def d_weighted(self, a, y, c):
        '''
        represents dJ/da

        Parameters

        a: Predicted output array of shape (batch, d)
        y: Actual output array of shape (batch, d)
        c: Weight factor array of shape (batch, 1)
        '''
        return 2 * c * (a - y)

    def log(self, a, y, epsilon=1e-12):
        '''
        Parameters

        a: Predicted output array of shape (batch, d)
        y: Actual output array of shape (batch, d)
        '''
        a = np.clip(a, epsilon, 1. - epsilon)
        return -y * np.log(a) - (1 - y) * np.log(1 - a)

    def d_log(self, a, y, epsilon=1e-12):
        '''
        represents dJ/da

        Parameters

        a: Predicted output array of shape (batch, d)
        y: Actual output array of shape (batch, d)
        '''
        a = np.clip(a, epsilon, 1. - epsilon)
        return -y / a + (1 - y) / (1 - a)

    def kl(self, a, y, epsilon=1e-12):
        '''
        Parameters

        a: Predicted output array of shape (batch, d)
        y: Actual output array of shape (batch, d)
        '''
        a = np.clip(a, epsilon, 1. - epsilon)
        return np.sum(y * np.log(y / a), axis=1)

    def d_kl(self, a, y, epsilon=1e-12):
        '''
        represents dJ/da

        Parameters

        a: Predicted output array of shape (batch, d)
        y: Actual output array of shape (batch, d)
        '''
        a = np.clip(a, epsilon, 1. - epsilon)
        return -y / a

    def exp(self, a, y):
        '''
        Parameters

        a: Predicted output array of shape (batch, d)
        y: Actual output array of shape (batch, d)
        '''
        return np.exp(-y * a)

    def d_exp(self, a, y):
        '''
        represents dJ/da

        Parameters

        a: Predicted output array of shape (batch, d)
        y: Actual output array of shape (batch, d)
        '''
        return -y * np.exp(-y * a)

    def hinge(self, a, y):
        '''
        Parameters

        a: Predicted output array of shape (batch, d)
        y: Actual output array of shape (batch, d)
        '''
        return np.maximum(0, 1 - y * a)

    def d_hinge(self, a, y):
        '''
        represents dJ/da

        Parameters

        a: Predicted output array of shape (batch, d)
        y: Actual output array of shape (batch, d)
        '''
        return np.where(y * a >= 1, 0, -y)

    def huber(self, a, y, delta):
        '''
        Parameters

        a: Predicted output array of shape (batch, d)
        y: Actual output array of shape (batch, d)
        delta: Hyperparameter for sensitivity to outliers
        '''
        error = a - y
        return np.where(np.abs(error) <= delta, 0.5 * error**2, delta * (np.abs(error) - 0.5 * delta))

    def d_huber(self, a, y, delta):
        '''
        represents dJ/da

        Parameters

        a: Predicted output array of shape (batch, d)
        y: Actual output array of shape (batch, d)
        delta: Hyperparameter for sensitivity to outliers
        '''
        error = a - y
        return np.where(np.abs(error) <= delta, error, delta * np.sign(error))


    def get_cost(self, a, y, *args):
        '''
        Parameters

        a: Predicted output array of shape (batch, d)
        y: Actual output array of shape (batch, d)
        '''
        match self.cost_type:
            case 'mse': return self.mse(a, y)
            case 'cross-entropy': return self.cross_entropy(a, y)
            case 'weighted': return self.weighted(a, y, *args)
            case 'log': return self.log(a, y, *args)
            case 'kullback-leibler': return self.kl(a, y, *args)
            case 'exp': return self.exp(a, y)
            case 'hinge': return self.hinge(a, y)
            case 'huber': return self.huber(a, y, *args)
            case 'binary_crossentropy': return self.binary_crossentropy(a, y, *args)
            case _:   raise ValueError("Valid cost functions are only: "+ ", ".join(self.names))

    def get_d_cost(self, a, y, *args):
        '''
        Parameters

        a: Predicted output array of shape (batch, d)
        y: Actual output array of shape (batch, d)
        '''
        match self.cost_type:
            case 'mse': return self.d_mse(a, y)
            case 'cross-entropy': return self.d_cross_entropy(a, y)
            case 'weighted': return self.d_weighted(a, y, *args)
            case 'log': return self.d_log(a, y, *args)
            case 'kullback-leibler': return self.d_kl(a, y, *args)
            case 'exp': return self.d_exp(a, y)
            case 'hinge': return self.d_hinge(a, y)
            case 'huber': return self.d_huber(a, y, *args)
            case 'binary_crossentropy': return self.d_binary_crossentropy(a, y, *args)
            case _:   raise ValueError("Valid cost functions are only: "+ ", ".join(self.names))








# In[ ] Optimizer:


class Optimizer:

    def __init__(self, optimizer_type=None, shape_W=None, shape_b=None,
                 momentum1=0.9, momentum2=0.999, epsilon=1e-8):
        '''
        Parameters

        momentum1: float hyperparameter >= 0 that accelerates gradient descent in the relevant
                   direction and dampens oscillations. Defaults to 0, i.e., vanilla gradient descent.
                   Also used in RMSProp
        momentum2: used in Adam only
        optimizer_type: type of optimizer
                        available options are 'gd', 'sgd' (This also includes momentum), 'adam', and 'rmsprop'
        shape_W: Shape of the weight matrix W/ Kernel K
        shape_b: Shape of the bias matrix b
        epsilon: parameter used in RMSProp and Adam to avoid division by zero error
        '''

        if optimizer_type is None:
            self.optimizer_type = 'adam'
        else:
            self.optimizer_type = optimizer_type

        self.momentum1 = momentum1
        self.momentum2 = momentum2
        self.epsilon = epsilon

        self.vdW = np.zeros(shape_W)
        self.vdb = np.zeros(shape_b)

        self.SdW = np.zeros(shape_W)
        self.Sdb = np.zeros(shape_b)
        
        if(optimizer_type.lower() == 'rprop'):
            self.delta_W = np.ones(shape_W)
            self.delta_b= np.ones(shape_b)
            self.eta_plus = 1.2
            self.eta_minus = 0.5
            self.delta_max = 50
            self.delta_min = 1e-6

    '''
        dW: gradient of Weight W for iteration k
        db: gradient of bias b for iteration k
        k: iteration number
    '''
    
    def GD(self, dW, db, k = None): #Gradient Descent
        
        return dW, db

    def SGD(self, dW, db, k = None): #Stochastic Gradient Descent (with momentum)
        self.vdW = self.momentum1*self.vdW + (1-self.momentum1)*dW
        self.vdb = self.momentum1*self.vdb + (1-self.momentum1)*db

        return self.vdW, self.vdb

    def RMSProp(self, dW, db, k = None):
        self.SdW = self.momentum2*self.SdW + (1-self.momentum2)*(dW**2)
        self.Sdb = self.momentum2*self.Sdb + (1-self.momentum2)*(db**2)

        den_W = np.sqrt(self.SdW) + self.epsilon
        den_b = np.sqrt(self.Sdb) + self.epsilon

        return dW/den_W, db/den_b

    def Adam(self, dW, db, k):
        '''
        dW: gradient of Weight W for iteration k
        db: gradient of bias b for iteration k
        k: iteration number
        '''
        # momentum
        self.vdW = self.momentum1*self.vdW + (1-self.momentum1)*dW
        self.vdb = self.momentum1*self.vdb + (1-self.momentum1)*db

        # rmsprop
        self.SdW = self.momentum2*self.SdW + (1-self.momentum2)*(dW**2)
        self.Sdb = self.momentum2*self.Sdb + (1-self.momentum2)*(db**2)

        # correction
        if k>1:
            vdW_h = self.vdW / (1-(self.momentum1**k))
            vdb_h = self.vdb / (1-(self.momentum1**k))
            SdW_h = self.SdW / (1-(self.momentum2**k))
            Sdb_h = self.Sdb / (1-(self.momentum2**k))
        else:
            vdW_h = self.vdW
            vdb_h = self.vdb
            SdW_h = self.SdW
            Sdb_h = self.Sdb

        den_W = np.sqrt(SdW_h) + self.epsilon
        den_b = np.sqrt(Sdb_h) + self.epsilon

        return vdW_h/den_W, vdb_h/den_b
    
    def Adagrad(self, dW, db, k):
        # Update the sum of squares of gradients
        self.SdW += dW**2
        self.Sdb += db**2

        # Calculate the denominator
        den_W = np.sqrt(self.SdW) + self.epsilon
        den_b = np.sqrt(self.Sdb) + self.epsilon

        # Calculate the updated gradients
        return dW/den_W, db/den_b

    def Adadelta(self, dW, db, k):
        # Update the sum of squares of gradients
        self.SdW = self.momentum2*self.SdW + (1-self.momentum2)*(dW**2)
        self.Sdb = self.momentum2*self.Sdb + (1-self.momentum2)*(db**2)

        # Calculate the numerator
        num_W = np.sqrt(self.SdW_h + self.epsilon)*dW
        num_b = np.sqrt(self.Sdb_h + self.epsilon)*db

        # Calculate the denominator
        den_W = np.sqrt(self.SdW) + self.epsilon
        den_b = np.sqrt(self.Sdb) + self.epsilon

        # Update the sum of squares of deltas
        self.SdW_h = self.momentum1*self.SdW_h + (1-self.momentum1)*(num_W**2)
        self.Sdb_h = self.momentum1*self.Sdb_h + (1-self.momentum1)*(num_b**2)

        # Calculate the updated gradients
        return num_W/den_W, num_b/den_b

    def NAG(self, dW, db, k): #Nesterov_Accelerated_Gradient
        # Store the previous velocity
        vdW_prev = self.vdW
        vdb_prev = self.vdb

        # Update the velocity
        self.vdW = self.momentum1*self.vdW - self.learning_rate*dW
        self.vdb = self.momentum1*self.vdb - self.learning_rate*db

        # Calculate the corrected velocity
        vdW_corr = self.momentum1*self.vdW + (1-self.momentum1)*vdW_prev
        vdb_corr = self.momentum1*self.vdb + (1-self.momentum1)*vdb_prev

        # Calculate the denominator
        den_W = np.sqrt(self.SdW) + self.epsilon
        den_b = np.sqrt(self.Sdb) + self.epsilon

        # Calculate the updated gradients
        return vdW_corr/den_W, vdb_corr/den_b

    def AdaMax(self, dW, db, k):
        # Update the sum of powers of gradients
        self.SdW = self.beta2*self.SdW + (1-self.beta2)*(dW**2)
        self.Sdb = self.beta2*self.Sdb + (1-self.beta2)*(db**2)

        # Calculate the numerator
        num_W = self.learning_rate*dW/(np.sqrt(self.SdW) + self.epsilon)
        num_b = self.learning_rate*db/(np.sqrt(self.Sdb) + self.epsilon)

        # Calculate the updated gradients
        return num_W, num_b

    def Nadam(self, dW, db, k, delta_min, delta_max):
        # Update the momentum
        self.vdW = self.momentum1*self.vdW + (1-self.momentum1)*dW
        self.vdb = self.momentum1*self.vdb + (1-self.momentum1)*db

        # Update the sum of squares of gradients
        self.SdW = self.momentum2*self.SdW + (1-self.momentum2)*(dW**2)
        self.Sdb = self.momentum2*self.Sdb + (1-self.momentum2)*(db**2)

        # Calculate the corrected momentum
        vdW_corr = self.vdW/(1-(self.momentum1**k))
        vdb_corr = self.vdb/(1-(self.momentum1**k))

        # Calculate the corrected sum of squares of gradients
        SdW_corr = self.SdW/(1-(self.momentum2**k))
        Sdb_corr = self.Sdb/(1-(self.momentum2**k))

        # Calculate the numerator
        num_W = self.learning_rate*(self.momentum1*vdW_corr + ((1-self.momentum1)/(1-(self.momentum1**k)))*dW)/(np.sqrt(SdW_corr) + self.epsilon)
        num_b = self.learning_rate*(self.momentum1*vdb_corr + ((1-self.momentum1)/(1-(self.momentum1**k)))*db)/(np.sqrt(Sdb_corr) + self.epsilon)
        return num_W, num_b
        
    def Rprop(self, dW, db, k):
        self.delta_W = np.clip(self.delta_W * np.where(np.sign(dW) == np.sign(self.delta_W), self.eta_plus, self.eta_minus), delta_min, delta_max)
        self.delta_b = np.clip(self.delta_b * np.where(np.sign(db) == np.sign(self.delta_b), self.eta_plus, self.eta_minus), delta_min, delta_max)
        nW = -np.sign(dW) * self.delta_W
        nb = -np.sign(db) * self.delta_b
        return nW, nb

    def get_optimization(self, dW, db, k, *args):
        match self.optimizer_type:
            case 'gd': return self.GD(dW, db, k)
            case 'sgd': return self.SGD(dW, db, k)
            case 'rmsprop': return self.RMSProp(dW, db, k)
            case 'adam': return self.Adam(dW, db, k)
            case 'adagrad': return self.Adagrad(dW, db, k)
            case 'adadelta': return self.Adadelta(dW, db, k)
            case 'nag': return self.NAG(dW, db, k)
            case 'adamax': return self.AdaMax(dW, db, k)
            case 'nadam': return self.Nadam(dW, db, k, *args)
            case 'rprop': return self.Rprop(dW, db, k)
            case _: raise ValueError("Valid optimizer options are only 'gd', 'sgd', 'rmsprop', 'adam', 'adagrad', 'adadelta', 'nag', 'adamax', 'nadam' and 'rprop'.")
        


# In[ ] Learning Rate Decay:


class LearningRateDecay:

    def __init__(self):
        pass

    def constant(self, t, learnrate_0):
        '''
        t: iteration
        learnrate_0: initial learning rate
        '''
        return learnrate_0

    def time_decay(self, t, learnrate_0, k):
        '''
        learnrate_0: initial learning rate
        k: Decay rate
        t: iteration number
        '''
        learnrate = learnrate_0 /(1+(k*t))
        return learnrate

    def step_decay(self, t, learnrate_0, F, D):
        '''
        learnrate_0: initial learning rate
        F: factor value controlling the rate in which the learning date drops
        D: “Drop every” iteration
        t: current iteration
        '''
        mult = F**np.floor((1+t)/D)
        learnrate = learnrate_0 * mult
        return learnrate

    def exponential_decay(self, t, learnrate_0, k):
        '''
        learnrate_0: initial learning rate
        k: Exponential Decay rate
        t: iteration number
        '''
        learnrate = learnrate_0 * np.exp(-k*t)
        return learnrate
    
    def inverse_time_decay(self, t, learnrate_0, k):
        '''
        learnrate_0: initial learning rate
        k: Decay rate
        t: iteration number
        '''
        learnrate = learnrate_0 / (1 + k * t)
        return learnrate

    def natural_exp_decay(self, t, learnrate_0, k):
        '''
        learnrate_0: initial learning rate
        k: Decay rate
        t: iteration number
        '''
        learnrate = learnrate_0 * np.exp(-k * t)
        return learnrate

    def piecewise_constant_decay(self, t, learnrate_0, boundaries, values):
        '''
        learnrate_0: initial learning rate
        t: iteration number
        boundaries: list of iteration numbers at which the learning rate changes
        values: list of learning rates corresponding to the intervals defined by boundaries
        '''
        for b, v in zip(boundaries, values):
            if t < b:
                return v
        return values[-1]
    
    def polynomial_decay(self, t, learnrate_0, power=1):
        '''
        learnrate_0: initial learning rate
        power: exponent of the polynomial
        t: iteration number
        '''
        learnrate = learnrate_0 / (1 + t)**power
        return learnrate

    def cosine_decay(self, t, learnrate_0, T_max):
        '''
        learnrate_0: initial learning rate
        T_max: maximum number of iterations
        t: iteration number
        '''
        learnrate = learnrate_0 * 0.5 * (1 + np.cos(np.pi * t / T_max))
        return learnrate

    def linear_cosine_decay(self, t, learnrate_0, T_max):
        '''
        learnrate_0: initial learning rate
        T_max: maximum number of iterations
        t: iteration number
        '''
        alpha = t / T_max
        learnrate = (1 - alpha) * learnrate_0 * 0.5 * (1 + np.cos(np.pi * t / T_max)) + alpha * learnrate_0
        return learnrate
    
    def warmup_decay(self, t, learnrate_0, warmup_steps=10):
        '''
        learnrate_0: initial learning rate
        warmup_steps: number of initial iterations for which to use linearly increasing learning rate
        t: iteration number
        '''
        if t < warmup_steps:
            return learnrate_0 * (t / warmup_steps)
        else:
            return learnrate_0

    def cyclic_decay(self, t, learnrate_0, step_size, max_lr, mode='triangular'):
        '''
        learnrate_0: initial learning rate
        step_size: number of iterations in one half cycle
        max_lr: maximum learning rate
        mode: type of cyclic decay ('triangular', 'triangular2', 'exp_range')
        t: iteration number
        '''
        cycle = np.floor(1 + t / (2 * step_size))
        x = np.abs(t / step_size - 2 * cycle + 1)
        if mode == 'triangular':
            lr = learnrate_0 + (max_lr - learnrate_0) * np.maximum(0, (1 - x))
        elif mode == 'triangular2':
            lr = learnrate_0 + (max_lr - learnrate_0) * np.maximum(0, (1 - x)) / (2 ** (cycle - 1))
        elif mode == 'exp_range':
            lr = learnrate_0 + (max_lr - learnrate_0) * np.maximum(0, (1 - x)) * (0.99999 ** t)
        return lr

    def sqrt_decay(self, t, learnrate_0, k=1):
        '''
        learnrate_0: initial learning rate
        k: Decay rate
        t: iteration number
        '''
        learnrate = learnrate_0 / np.sqrt(k * t + 1)
        return learnrate

    def power_decay(self, t, learnrate_0, k=0.1, power=0.75):
        '''
        learnrate_0: initial learning rate
        k: Decay rate
        power: power to which iteration number is raised
        t: iteration number
        '''
        learnrate = learnrate_0 / (1 + k * t)**power
        return learnrate

    def staircase_decay(self, t, learnrate_0, drop_rate=0.5, epochs_drop=10.0):
        '''
        learnrate_0: initial learning rate
        drop_rate: factor by which learning rate is reduced at each stage
        epochs_drop: number of epochs after which learning rate drops
        t: iteration number
        '''
        learnrate = learnrate_0 * (drop_rate ** np.floor((1+t) / epochs_drop))
        return learnrate

    def cosine_annealing_decay(self, t, learnrate_0, T_max, M):
        '''
        learnrate_0: initial learning rate
        T_max: maximum number of iterations
        M: number of cycles
        t: iteration number
        '''
        learnrate = learnrate_0 / 2 * (np.cos(np.pi * (t % (T_max // M)) / (T_max // M)) + 1)
        return learnrate

    def burnin_decay(self, t, learnrate_0, burnin, learnrate_burnin):
        '''
        learnrate_0: initial learning rate
        burnin: number of initial iterations for which to use burnin learning rate
        learnrate_burnin: learning rate used during burnin period
        t: iteration number
        '''
        if t < burnin:
            return learnrate_burnin
        else:
            return learnrate_0

    def warm_restart_decay(self, t, learnrate_0, T_0, T_mult):
        '''
        learnrate_0: initial learning rate
        T_0: initial number of iterations
        T_mult: factor by which number of iterations increases after each restart
        t: iteration number
        '''
        T_i = T_0
        while t >= T_i:
            t = t - T_i
            T_i = T_i * T_mult
        learnrate = learnrate_0 * 0.5 * (1 + np.cos(np.pi * t / T_i))
        return learnrate
    
    def decay_on_plateau(self, t, learnrate_0, min_lr, factor=0.1, patience=10, cooldown=0):
        '''
        learnrate_0: initial learning rate
        min_lr: minimum learning rate
        factor: factor by which the learning rate is reduced
        patience: number of epochs with no improvement after which learning rate will be reduced
        cooldown: number of epochs to wait before resuming normal operation after lr has been reduced
        t: iteration number
        '''
        if t < cooldown:
            return learnrate_0
        elif t % patience == 0:
            return max(learnrate_0 * factor, min_lr)
        else:
            return learnrate_0

    def one_cycle_policy(self, t, learnrate_max, stepsize, base_lr=None):
        '''
        learnrate_max: maximum learning rate
        stepsize: number of iterations in one half cycle
        base_lr: initial learning rate
        t: iteration number
        '''
        if base_lr is None:
            base_lr = learnrate_max / 10
        if t <= stepsize:
            return base_lr + (learnrate_max - base_lr) * t / stepsize
        elif t <= 2 * stepsize:
            return learnrate_max - (learnrate_max - base_lr) * (t - stepsize) / stepsize
        else:
            return base_lr
    
    def cosine_decay_restarts(self, t, learnrate_0, T_0, T_mult=2):
        '''
        learnrate_0: initial learning rate
        T_0: initial number of iterations
        T_mult: factor by which number of iterations increases after each restart
        t: iteration number
        '''
        T_i = T_0
        while t >= T_i:
            t = t - T_i
            T_i = T_i * T_mult
        learnrate = learnrate_0 * 0.5 * (1 + np.cos(np.pi * t / T_i))
        return learnrate

    def triangular_learning_rate(self, t, learnrate_low, learnrate_high, T):
        '''
        learnrate_low: lower bound for learning rate
        learnrate_high: upper bound for learning rate
        T: total number of iterations
        t: iteration number
        '''
        cycle = np.floor(1 + t / (2 * T))
        x = np.abs(t / T - 2 * cycle + 1)
        learnrate = learnrate_low + (learnrate_high - learnrate_low) * np.maximum(0, (1 - x))
        return learnrate

    def triangular2_learning_rate(self, t, learnrate_low, learnrate_high, T):
        '''
        learnrate_low: lower bound for learning rate
        learnrate_high: upper bound for learning rate
        T: total number of iterations
        t: iteration number
        '''
        cycle = np.floor(1 + t / (2 * T))
        x = np.abs(t / T - 2 * cycle + 1)
        learnrate = learnrate_low + (learnrate_high - learnrate_low) * np.maximum(0, (1 - x)) / (2 ** (cycle - 1))
        return learnrate
    
    def exp_range_decay(self, t, learnrate_0, gamma, step_size):
        '''
        learnrate_0: initial learning rate
        gamma: decay rate
        step_size: number of iterations in one half cycle
        t: iteration number
        '''
        cycle = np.floor(1 + t / (2 * step_size))
        x = np.abs(t / step_size - 2 * cycle + 1)
        learnrate = learnrate_0 * (gamma**(t)) * np.maximum(0, (1 - x))
        return learnrate

    def triangular_lr_decay(self, t, learnrate_low, learnrate_high, T):
        '''
        learnrate_low: lower bound for learning rate
        learnrate_high: upper bound for learning rate
        T: total number of iterations
        t: iteration number
        '''
        cycle = np.floor(1 + t / (2 * T))
        x = np.abs(t / T - 2 * cycle + 1)
        learnrate = learnrate_low + (learnrate_high - learnrate_low) * np.maximum(0, (1 - x))
        return learnrate

    def triangular2_lr_decay(self, t, learnrate_low, learnrate_high, T):
        '''
        learnrate_low: lower bound for learning rate
        learnrate_high: upper bound for learning rate
        T: total number of iterations
        t: iteration number
        '''
        cycle = np.floor(1 + t / (2 * T))
        x = np.abs(t / T - 2 * cycle + 1)
        learnrate = learnrate_low + (learnrate_high - learnrate_low) * np.maximum(0, (1 - x)) / (2 ** (cycle - 1))
        return learnrate







# In[ ]Utility:


class Utility:

    def __init__(self):
        pass

    def label_encoding(self, Y):
        '''
        Parameters:
        Y: (batch,d) shape matrix with categorical data
        Return
        result: label encoded data of 𝑌
        idx_list: list of the dictionaries containing the unique values
                  of the columns and their mapping to the integer.
        '''
        idx_list = []
        result = []
        for col in range(Y.shape[1]):
            indexes = {val: idx for idx, val in enumerate(np.unique(Y[:, col]))}
            result.append([indexes[s] for s in Y[:, col]])
            idx_list.append(indexes)
        return np.array(result).T, idx_list

    def onehot(self, X):
        '''
        Parameters:
        X: 1D array of labels of length "batch"
        Return
        X_onehot: (batch,d) one hot encoded matrix (one-hot of X)
                  (where d is the number of unique values in X)
        indexes: dictionary containing the unique values of X and their mapping to the integer column
        '''
        indexes = {val: idx for idx, val in enumerate(np.unique(X))}
        y = np.array([indexes[s] for s in X])
        X_onehot = np.zeros((y.size, len(indexes)))
        X_onehot[np.arange(y.size), y] = 1
        return X_onehot, indexes

    def minmax(self, X, min_X=None, max_X=None):
        if min_X is None:
            min_X = np.min(X, axis=0)
        if max_X is None:
            max_X = np.max(X, axis=0)
        Z = (X - min_X) / (max_X - min_X)
        return Z, min_X, max_X

    def standardize(self, X, mu=None, std=None):
        if mu is None:
            mu = np.mean(X, axis=0)
        if std is None:
            std = np.std(X, axis=0)
        Z = (X - mu) / std
        return Z, mu, std

    def inv_standardize(self, Z, mu, std):
        X = Z*std + mu
        return X

    def train_test_split(self, X, y, test_ratio=0.2, seed=None):
        if seed is not None:
            np.random.seed(seed)
        train_ratio = 1-test_ratio
        indices = np.random.permutation(X.shape[0])
        train_idx, test_idx = indices[:int(train_ratio*len(X))], indices[int(train_ratio*len(X)):]
        X_train, X_test = X[train_idx,:], X[test_idx,:]
        y_train, y_test = y[train_idx], y[test_idx]
        return X_train, X_test, y_train, y_test








# In[ ] Weights Initializer :


class Weights_initializer:

    def __init__(self, shape, initializer_type=None, seed=None):
        '''
        Parameters
        shape: Shape of the weight matrix

        initializer_type: type of weight initializer
        available options are 'zeros', 'ones', 'random_normal', 'random_uniform',
        'he_normal', 'xavier_normal' and 'glorot_normal'
        '''
        self.shape = shape
        if initializer_type is None:
            self.initializer_type = "he_normal"
        else:
            self.initializer_type = initializer_type
        self.seed = seed

    def zeros_initializer(self):
        if self.seed is not None:
            np.random.seed(self.seed)
        return np.zeros(self.shape)

    def ones_initializer(self):
        if self.seed is not None:
            np.random.seed(self.seed)
        return np.ones(self.shape)

    def random_normal_initializer(self):
        if self.seed is not None:
            np.random.seed(self.seed)
        return np.random.normal(size=self.shape)

    def random_uniform_initializer(self):
        if self.seed is not None:
            np.random.seed(self.seed)
        return np.random.uniform(size=self.shape)

    def he_initializer(self):
        if self.seed is not None:
            np.random.seed(self.seed)
        try:
            F, kernelC, kernelH, kernelW = self.shape
        except:
            kernelH, kernelW = self.shape
        return np.random.randn(*self.shape) * np.sqrt(2/kernelH)

    def xavier_initializer(self):
        '''
        shape: Shape of the Kernel matrix.
        '''
        if self.seed is not None:
            np.random.seed(self.seed)
        try:
            F, kernelC, kernelH, kernelW = self.shape
        except:
            kernelH, kernelW = self.shape
        return np.random.randn(*self.shape) * np.sqrt(1/kernelH)

    def glorot_initializer(self):
        '''
        shape: Shape of the weight matrix.
        '''
        if self.seed is not None:
            np.random.seed(self.seed)
        try:
            F, kernelC, kernelH, kernelW = self.shape
        except:
            kernelH, kernelW = self.shape
        return np.random.randn(*self.shape) * np.sqrt(2/(kernelH+kernelW))

    def lecun_normal_initializer(self):
        if self.seed is not None:
            np.random.seed(self.seed)
        return np.random.randn(*self.shape) * np.sqrt(1. / self.shape[1])

    def lecun_uniform_initializer(self):
        if self.seed is not None:
            np.random.seed(self.seed)
        limit = np.sqrt(3. / self.shape[1])
        return np.random.uniform(-limit, limit, size=self.shape)

    def glorot_uniform_initializer(self):
        if self.seed is not None:
            np.random.seed(self.seed)
        limit = np.sqrt(6. / (self.shape[0] + self.shape[1]))
        return np.random.uniform(-limit, limit, size=self.shape)
    
    def he_uniform_initializer(self):
        if self.seed is not None:
            np.random.seed(self.seed)
        limit = np.sqrt(6. / self.shape[1])
        return np.random.uniform(-limit, limit, size=self.shape)

    def xavier_uniform_initializer(self):
        if self.seed is not None:
            np.random.seed(self.seed)
        limit = np.sqrt(3. / (self.shape[0] + self.shape[1]))
        return np.random.uniform(-limit, limit, size=self.shape)
    
    def constant_initializer(self, constant=0):
        if self.seed is not None:
            np.random.seed(self.seed)
        return np.full(self.shape, constant)

    def orthogonal_initializer(self, gain=1.0):
        if self.seed is not None:
            np.random.seed(self.seed)
        flat_shape = (self.shape[0], np.prod(self.shape[1:]))
        a = np.random.normal(0.0, 1.0, flat_shape)
        u, _, v = np.linalg.svd(a, full_matrices=False)
        q = u if u.shape == flat_shape else v
        return gain * q.reshape(self.shape)
    
    def variance_scaling_initializer(self, scale=1.0, mode='fan_in', distribution='normal'):
        if self.seed is not None:
            np.random.seed(self.seed)
        fan_in, fan_out = self.shape[0], self.shape[1]
        if mode == 'fan_avg':
            fan_avg = (fan_in + fan_out) / 2.0
            scale /= max(1., fan_avg)
        elif mode == 'fan_in':
            scale /= max(1., fan_in)
        elif mode == 'fan_out':
            scale /= max(1., fan_out)
        else:
            raise ValueError('Invalid mode for variance scaling initializer: %s.' % mode)
        if distribution == 'normal':
            stddev = np.sqrt(scale)
            return np.random.normal(0., stddev, size=self.shape)
        elif distribution == 'uniform':
            limit = np.sqrt(3. * scale)
            return np.random.uniform(-limit, limit, size=self.shape)
        else:
            raise ValueError('Invalid distribution for variance scaling initializer: %s.' % distribution)
    
    def truncated_normal_initializer(self, mean=0.0, stddev=0.05):
        if self.seed is not None:
            np.random.seed(self.seed)
        values = np.random.normal(loc=mean, scale=stddev, size=self.shape)
        return np.clip(values, mean - 2*stddev, mean + 2*stddev)

    def identity_initializer(self, gain=1.0):
        if self.seed is not None:
            np.random.seed(self.seed)
        if len(self.shape) != 2 or self.shape[0] != self.shape[1]:
            raise ValueError('Identity matrix initializer can only be used for 2D square matrices.')
        else:
            return np.eye(self.shape[0]) * gain
        
    def uniform_initializer(self, minval=0, maxval=None):
        if self.seed is not None:
            np.random.seed(self.seed)
        return np.random.uniform(minval, maxval, size=self.shape)
    
    def normal_initializer(self, mean=0.0, stddev=1.0):
        if self.seed is not None:
            np.random.seed(self.seed)
        return np.random.normal(loc=mean, scale=stddev, size=self.shape)
    
    def get_initializer(self, *args):
        match self.initializer_type.lower():
            case 'zeros': return self.zeros_initializer(*args)
            case 'ones': return self.ones_initializer(*args)
            case 'random_normal': return self.random_normal_initializer(*args)
            case 'random_uniform': return self.random_uniform_initializer(*args)
            case 'he_normal': return self.he_initializer(*args)
            case 'glorot_normal': return self.glorot_initializer(*args)
            case 'xavier_normal': return self.xavier_initializer()
            case 'lecun_normal': return self.lecun_normal_initializer(*args)
            case 'lecun_uniform': return self.lecun_uniform_initializer(*args)
            case 'glorot_uniform': return self.glorot_uniform_initializer(*args)
            case 'he_uniform': return self.he_uniform_initializer(*args)
            case 'xavier_uniform': return self.xavier_uniform_initializer(*args)
            case 'constant': return self.constant_initializer(*args)
            case 'orthogonal': return self.orthogonal_initializer(*args)
            case 'variance_scaling': return self.variance_scaling_initializer(*args)
            case 'truncated_normal': return self.truncated_normal_initializer(*args)
            case 'identity': return self.identity_initializer(*args)
            case 'uniform': return self.uniform_initializer(*args)
            case 'normal': return self.normal_initializer(*args)
            case _: raise ValueError("Valid initializer options are 'zeros', 'ones', 'random_normal', 'random_uniform', 'he_normal', 'xavier_normal', and 'glorot_normal', ...")
       
