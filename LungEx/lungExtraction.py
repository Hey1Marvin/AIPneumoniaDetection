'''
Create Classes for U-NET via tensorflow
then train one model with the JRST- Dataset to extract the lung, hart usw.
'''

#import necessary modules
import os
import cv2
import torch
import fnmatch
import torchvision
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.optim as optim
import torchvision.transforms as transforms
from imageio import imread
from getpass import getpass
from torch.nn   import functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, models, transforms

#Global Variables 
IMG_SIZE = 256
SIZE = 256

def num2fixstr(x,d):
    # example num2fixstr(2,5) returns '00002'
    # example num2fixstr(19,3) returns '019'
    st = '%0*d' % (d,x)
    return st
        

#classes for the UNet
class Unet(nn.Module):

    # Implementation of U-Net inhereting from nn.Module
    # Implementation with batchnorm

    def __init__(self, n, batch_size=10, chan_out=1, batchnorm=False):

        super().__init__()
        
        self.conv_1 = self.conv_layer(n, 0, batchnorm=batchnorm)
        self.conv_2 = self.conv_layer(n, 1, batchnorm=batchnorm)
        self.conv_3 = self.conv_layer(n, 2, batchnorm=batchnorm)
        self.conv_4 = self.conv_layer(n, 3, batchnorm=batchnorm)
        self.conv_5 = self.conv_layer(n, 4, batchnorm=batchnorm)
        self.conv_6 = self.conv_layer(n, 5, batchnorm=batchnorm)
        self.conv_7 = self.conv_layer(n, 6, batchnorm=batchnorm)
        self.conv_8 = self.conv_layer(n, 7, batchnorm=batchnorm)
        self.conv_9 = self.conv_layer(n, 8, batchnorm=batchnorm)

        self.conv_A = nn.Sequential( # output layer with Sigmoid function
            nn.Conv2d(in_channels=n, out_channels=chan_out, kernel_size=1, padding='same'),
            nn.Sigmoid()
        )

        self.up_6 = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels=16*n, out_channels=8*n, kernel_size=2, padding='same'),
            nn.ReLU()
        )

        self.up_7 = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels=8*n, out_channels=4*n, kernel_size=2, padding='same'),
            nn.ReLU()
        )

        self.up_8 = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels=4*n, out_channels=2*n, kernel_size=2, padding='same'),
            nn.ReLU()
        )

        self.up_9 = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels=2*n, out_channels=n, kernel_size=2, padding='same'),
            nn.ReLU()
        )

    #implementation of the convoltional blocks for the down path 
    def conv_layer(n, l_n, batchnorm=False):
        if l_n == 0:
            in_  = 1
            out_ = n
        elif l_n < 5:
            in_  = int(n * (2 ** (l_n - 1)))
            out_ = int(n * (2 ** l_n))
        else:
            in_  = int(n * (2 ** (9 - l_n) ))
            out_ = int(n * (2 ** (9 - l_n - 1)))

        if batchnorm:
            # apply Batchnorm after each ReLU
            conv = nn.Sequential(
                nn.Conv2d(in_channels=in_, out_channels=out_, kernel_size=3, padding='same'),
                nn.ReLU(),
                nn.BatchNorm2d(out_),
                nn.Conv2d(in_channels=out_, out_channels=out_, kernel_size=3, padding='same'),
                nn.ReLU(),
                nn.BatchNorm2d(out_),
            )
        else:
            # implementation without Batchnorm
            conv = nn.Sequential(
                nn.Conv2d(in_channels=in_, out_channels=out_, kernel_size=3, padding='same'),
                nn.ReLU(),
                nn.Conv2d(in_channels=out_, out_channels=out_, kernel_size=3, padding='same'),
                nn.ReLU(),
            )
            return conv


    def forward(self, input_):
        conv_1 = self.conv_1(input_)
        pool_1 = nn.MaxPool2d(kernel_size=(2,2))(conv_1)

        conv_2 = self.conv_2(pool_1)
        pool_2 = nn.MaxPool2d(kernel_size=(2,2))(conv_2)

        conv_3 = self.conv_3(pool_2)
        pool_3 = nn.MaxPool2d(kernel_size=(2,2))(conv_3)

        conv_4 = self.conv_4(pool_3)
        pool_4 = nn.MaxPool2d(kernel_size=(2,2))(conv_4)

        conv_5 = self.conv_5(pool_4)

        up_6    = self.up_6(conv_5)
        merge_6 = torch.cat([conv_4, up_6], dim=1)
        conv_6  = self.conv_6(merge_6)

        up_7    = self.up_7(conv_6)
        merge_7 = torch.cat([conv_3, up_7], dim=1)
        conv_7  = self.conv_7(merge_7)

        up_8    = self.up_8(conv_7)
        merge_8 = torch.cat([conv_2, up_8], dim=1)
        conv_8  = self.conv_8(merge_8)

        up_9    = self.up_9(conv_8)
        merge_9 = torch.cat([conv_1, up_9], dim=1)
        conv_9  = self.conv_9(merge_9)

        return self.conv_A(conv_9)



######## -------- Losses
class BCELoss2d(nn.Module):
    # calculater the binary loss for a 2D map
    # Fuente: https://github.com/hiyouga/Image-Segmentation-PyTorch/blob/master/loss_func.py
    def __init__(self):
        super(BCELoss2d, self).__init__()
        self.bce_loss = nn.BCELoss()
        
    def forward(self, predict, target):
        predict = predict.view(-1)
        target = target.view(-1)
        return self.bce_loss(predict, target)

class BCELoss3d(nn.Module):
    # Similar a anterior pero calcula BCE loss para clases por separado
    def __init__(self, classes):
        super(BCELoss3d, self).__init__()
        self.bce_loss = nn.BCELoss()
        self.classes = classes
        
    def forward(self, predict, target):
        loss = 0.
        for x in range(self.classes):
            loss += self.bce_loss(predict[:, x, :, :].flatten(), target[:, x, :, :].flatten())
        return loss / self.classes

def dice_loss(input, target):
    # Dice Loss normal para caso de una clase única
    # También funciona multiclase, pero no considera clases separadas
    # Fuente: https://github.com/pytorch/pytorch/issues/1249
    smooth = 1.

    iflat = input.view(-1)
    tflat = target.view(-1)
    intersection = (iflat * tflat).sum()
        
    return 1 - ((2. * intersection + smooth) /
                (iflat.sum() + tflat.sum() + smooth))
    
def log_cosh_dice_loss(input, target):
    return torch.log(torch.cosh(dice_loss(input, target)))
        


#########--------- Training Methods
def train_epoch(model, iterator, optimizer, criterion, device, experiment, e):
    # Código usual para entrenar una época
    # Agregamos código para logging de pérdidas
    epoch_loss = 0
    model.train()
    for batch in iterator:
        optimizer.zero_grad()

        img, seg = batch
        img = img.to(device)
        seg = seg.to(device)

        predictions = model(img)

        loss = criterion(predictions, seg)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    if experiment is not None:
        # Usamos COMET para logging de pérdidas
        experiment.log_metric("train_loss", epoch_loss / len(iterator), step=e)
        
    return epoch_loss/len(iterator)


def eval_epoch(model, iterator, criterion, device, experiment, e):

    #initialize every epoch
    epoch_loss = 0

    #deactivating dropout layers
    model.eval()

    #deactivates autograd
    with torch.no_grad():
        for batch in iterator:
            
            img, seg = batch
            img = img.to(device)
            seg = seg.to(device)
                    
            #convert to 1d tensor
            predictions = model(img)
                    
            #compute loss
            loss = criterion(predictions, seg)
                    
            #keep track of loss
            epoch_loss += loss.item()

    if experiment is not None:
        # Usamos COMET para logging de pérdidas
        experiment.log_metric("val_loss", epoch_loss / len(iterator), step=e)
            
    return epoch_loss / len(iterator)


def train_epochs(model, data, criterion, optimizer, device, epochs, experiment, name, early=10):
    # Función que dado un modelo lo entrena N épocas
    # También se encarga de logging usando Comet
    best_val = float("Inf")

    data_train, data_val = data
    early_stop = 0

    for epoch in range(epochs):
    
        # Training
        train_loss = train_epoch(model, data_train, optimizer, criterion, device, experiment, epoch)

        # Validation
        val_loss = eval_epoch(model, data_val, criterion, device, experiment, epoch)

        st = f'Training Loss: {train_loss:.4f}'
        sv = f'Validation Loss: {val_loss:.4f}'
        print('['+num2fixstr(epoch+1,3)+'/'+num2fixstr(epochs,3)+'] ' + st + ' | '+sv)

        if val_loss < best_val:
            # Guardamos el modelo con mejor pérdida en validación
            print(f"Best model achieved, saving to {name}.pth")
            torch.save(model.state_dict(), f"{name}.pth")
            best_val = val_loss
            early_stop = 0
        else:
            early_stop += 1
            if early_stop >= early:
                print(f"Early model stop. Best validation loss: {best_val}")
                break

     

def define_train_unet(data, batch_size, device, name, epochs=150, pretrained=None, n=SIZE, 
                      loss="MSE", optimizer="Adam", lr=1e-4, experiment=False, bn=False, 
                      early=10, chan_out=1):
    exp = None
    if experiment:
        # Log experiment to COMET
        exp = Experiment(
        api_key      = getpass(prompt='Key: '), # Secret
        project_name = "",
        workspace    = "",
        )

        # Parameters to log
        params = {
            "Model Size": n,
            "epochs": epochs,
            "loss": loss,
            "optimizer": optimizer,
            "learning rate": lr,
            "name": name
        }

        exp.log_parameters(params)

    if pretrained is not None:
        # Continuamos entrenando un modelo preentrenado
        model = pretrained
    else:
        # Instanciamos el modelo de UNet
        model = Unet(n, batchnorm=bn, chan_out=chan_out).to(device)

    # Probar con otras pérdidas
    if loss == "MSE":
        criterion = nn.MSELoss().to(device)
    elif loss == "CrossEntropy":
        criterion = BCELoss2d().to(device)
    elif loss == "Dice":
        criterion = dice_loss
    elif loss == "CrossEntropy_all":
        criterion = BCELoss3d(classes=3).to(device)
    elif loss == "log_cosh":
        criterion = log_cosh_dice_loss
    else:
        raise NotImplementedError(f"Loss not implemented: {loss}")

    # Distintos optimizadores
    if optimizer == "Adam":
        opti = optim.Adam(model.parameters(), lr=lr)
    else:
        raise NotImplementedError(f"Optimizer not implemented: {optimizer}")

    try:
        # Entrenamos el modelo
        train_epochs(model, data, criterion, opti, device, epochs, exp, name, early=early)
    except KeyboardInterrupt:
        if experiment:
            exp.end()

    if experiment:
        # End current experiment
        exp.end()

    return model



######-------- Loading the Traindata
def dirfiles(img_path, img_ext):
    img_names = fnmatch.filter(sorted(os.listdir(img_path)),img_ext)
    return img_names

def load_images(fpath, img_names, echo='off', nl=IMG_SIZE, nc=IMG_SIZE):
    # Cargamos las imágenes y las dejamos en un diccionario
    images = {}
    for name in img_names:
        st = fpath + os.sep + name
        # Extraemos las imágenes de tipo .IMG
        # https://forum.image.sc/t/how-to-open-the-images-from-jsrt-database-in-imagej/11317/7
        X = np.fromfile(st, dtype=">i2").reshape((2048, 2048)).astype('uint16')/65535.0
        X = cv2.resize(X, (nl, nc))
        # plt.imshow(X, cmap="gray")
        X = X.astype(np.double)
        images[name.split('.')[0]] = X
    return images

files  = dirfiles("All247images", "*.IMG")
images = load_images("All247images", files)

def load_combine_masks(fpath, middle, first_mask, second_mask, mask_dict, size=SIZE):
    files1 = f"{fpath}{os.sep}{middle}{os.sep}masks{os.sep}{first_mask}"
    files2 = f"{fpath}{os.sep}{middle}{os.sep}masks{os.sep}{second_mask}"

    names = dirfiles(files1, "*.gif")

    for name in names:
        # Tomamos ambas máscaras y las combinamos
        img1 = imread(files1 + os.sep + name)
        img1 = cv2.resize(img1, (size, size)) / 255
        img2 = imread(files2 + os.sep + name)
        img2 = cv2.resize(img2, (size, size)) / 255
        mask = np.maximum(img1, img2)
        mask_dict[name.split('.')[0]] = mask

def load_masks(fpath, test=False, mask_type="heart", size=SIZE):
    # Carga las máscaras de corazón o pulmónes
    masks = {}

    if test:
        # Carpeta de testing
        middle = "fold2"
    else:
        # Carpeta de training
        middle = "fold1"

    if mask_type == "lung":
        # Máscaras de ambos pulmones
        load_combine_masks(fpath, middle, "left lung", "right lung", masks)

    elif mask_type == "clavicle":
        # Máscaras de ambas clavículas
        load_combine_masks(fpath, middle, "left clavicle", "right clavicle", masks)

    else:
        # Buscamos las máscaras
        files  = f"{fpath}{os.sep}{middle}{os.sep}masks{os.sep}heart"
        names = dirfiles(files, "*.gif")

        for name in names:
            # Dejamos en el tamaño necesario y guardamos
            img = imread(files + os.sep + name)
            img = cv2.resize(img, (size, size)) / 255
            masks[name.split('.')[0]] = img

    return masks

##########--------- creating Pytorch Dataset
def trans(data):
    # Helper function
    return torch.unsqueeze(torch.from_numpy(data), dim=0).float()

def trans_(data):
    # Helper function
    return torch.from_numpy(data).float()
        

class JSRTDataset(Dataset):

    def __init__(self, images, test=False, mask="heart"):
        super().__init__()
        # Extraemos las máscaras necesarias
        # Asumimos que las imágenes se encuentran extraídas
        self.mask = mask

        if self.mask in ["heart", "all"]:
            heart_masks = load_masks("scratch", test=test)

        if self.mask in ["clavicles", "all"]:
            clav_masks = load_masks("scratch", mask_type="clavicle", test=test)

        if self.mask in ["lungs", "all"]:
            lung_masks  = load_masks("scratch", mask_type="lung", test=test)

        self.data = []

        if self.mask == "heart":
            self.add_data(images, heart_masks)
        elif self.mask == "clavicles":
            self.add_data(images, clav_masks)
        elif self.mask == "lungs":
            self.add_data(images, lung_masks)
        elif self.mask == "all":
            for img_name in heart_masks:
                mask_channels = [lung_masks[img_name], heart_masks[img_name], clav_masks[img_name]]
                mask_channels = [torch.from_numpy(ma).float() for ma in mask_channels]
                self.data.append((trans(images[img_name]), torch.stack(mask_channels)))

    def add_data(self, images, masks):
        for img_name in masks:
            self.data.append((trans(images[img_name]), trans(masks[img_name])))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, n):
        # Hacemos unsqueeze ya que sólo tenemos un canal de brillo
        # Float debido a que el modelo es float y los datos deben serlo
        item = self.data[n]
        return item[0], item[1]


##########------- Print the Metrics
def get_precision_recall(data, model):

    TP = 0
    FP = 0
    FN = 0

    model.eval()

    #deactivates autograd
    with torch.no_grad():
        for batch in data:
            
            img, seg = batch
            img = img.to(device)
                    
            #convert to 1d tensor
            predictions = (model(img).cpu().detach().numpy() > 0.5)*1
            seg = seg.cpu().detach().numpy()

            # Obtenemos y sumamos TP, FP y FN para todas las imágenes del batch

            TP += np.sum(np.multiply(predictions==1,seg==1))
            FP += np.sum(np.multiply(predictions==1,seg==0))
            FN += np.sum(np.multiply(predictions==0,seg==1))
    print(f"TP: {TP}, FP: {FP}, FN: {FN}")
    # Dados los TP, FP y FN totales obtenemos
    # Precision, Recall y F1-Score
    Pr = TP/(TP+FP)
    Re = TP/(TP+FN)
    F1 = 2 * (Pr * Re) / (Pr + Re)

    print('Testing:')
    print(f'Precision   = {Pr:.4f}')
    print(f'Recall      = {Re:.4f}')
    print(f'F1 Score    = {F1:.4f}')


def get_precision_recall_multiclass(data, model, classes=3):

    TP = np.zeros((classes))
    FP = np.zeros((classes))
    FN = np.zeros((classes))

    model.eval()

    #deactivates autograd
    with torch.no_grad():
        for batch in data:
            
            img, seg = batch
            img = img.to(device)
                    
            #convert to 1d tensor
            predictions = (model(img).cpu().detach().numpy() > 0.5)*1
            seg = seg.cpu().detach().numpy()

            # Obtenemos y sumamos TP, FP y FN para todas las imágenes del batch

            TP += np.sum(np.multiply(predictions==1,seg==1), axis=(0, 2, 3))
            FP += np.sum(np.multiply(predictions==1,seg==0), axis=(0, 2, 3))
            FN += np.sum(np.multiply(predictions==0,seg==1), axis=(0, 2, 3))

    # Dados los TP, FP y FN totales obtenemos
    # Precision, Recall y F1-Score
    Pr = TP/(TP+FP)
    Re = TP/(TP+FN)
    F1 = 2 * (Pr * Re) / (Pr + Re)

    # Macro stats
    mPr = np.mean(Pr)
    mRe = np.mean(Re)
    mF1 = 2 * (mPr * mRe) / (mPr + mRe)

    # Micro stats
    TP_t = np.sum(TP)
    FP_t = np.sum(FP)
    FN_t = np.sum(FN)

    miPr = TP_t/(TP_t+FP_t)
    miRe = TP_t/(TP_t+FN_t)
    miF1 = 2 * (miPr * miRe) / (miPr + miRe)

    print(f"Class Precisions: \nPulmones {Pr[0]:.4f}\t\tCorazón {Pr[1]:.4f}\t\tClavícula {Pr[2]:.4f}")
    print(f"Class Recalls: \nPulmones {Re[0]:.4f}\t\tCorazón {Re[1]:.4f}\t\tClavícula {Re[2]:.4f}")
    print(f"Class F1: \nPulmones {F1[0]:.4f}\t\tCorazón {F1[1]:.4f}\t\tClavícula {F1[2]:.4f}")

    print(f"\nPrecision (micro): {miPr:.4f}")
    print(f"Recall (micro): {miRe:.4f}")
    print(f"F1 (micro): {miF1:.4f}")

    print(f"\nPrecision (macro): {mPr:.4f}")
    print(f"Recall (macro): {mRe:.4f}")
    print(f"F1 (macro): {mF1:.4f}")

'''
#####------ Dataloader Heart
BATCH_SIZE = 5

# Test y train dataset
test_set_heart  = JSRTDataset(images, test=True, mask="heart")
train_set_heart = JSRTDataset(images, test=False, mask="heart")

# Creamos DataLoaders
testloader_heart  = DataLoader(dataset=test_set_heart, batch_size=BATCH_SIZE, shuffle=True)
trainloader_heart = DataLoader(dataset=train_set_heart, batch_size=BATCH_SIZE, shuffle=True)
'''
######---- Dataloader lung
BATCH_SIZE = 5

# Test y train dataset
test_set_lungs  = JSRTDataset(images, test=True, mask="lungs")
train_set_lungs = JSRTDataset(images, test=False, mask="lungs")

# Creamos DataLoaders
testloader_lungs  = DataLoader(dataset=test_set_lungs, batch_size=BATCH_SIZE, shuffle=True)
trainloader_lungs = DataLoader(dataset=train_set_lungs, batch_size=BATCH_SIZE, shuffle=True)

'''
####----- Dataloader Clavicles (Schlüsselbein)
BATCH_SIZE = 5

# Test y train dataset
test_set_clav  = JSRTDataset(images, test=True, mask="clavicles")
train_set_clav = JSRTDataset(images, test=False, mask="clavicles")

# Creamos DataLoaders
testloader_clav  = DataLoader(dataset=test_set_clav, batch_size=BATCH_SIZE, shuffle=True)
trainloader_clav = DataLoader(dataset=train_set_clav, batch_size=BATCH_SIZE, shuffle=True)


###--- Dataloader Multyclass
BATCH_SIZE = 5

# Test y train dataset
test_set_all  = JSRTDataset(images, test=True, mask="all")
train_set_all = JSRTDataset(images, test=False, mask="all")

# Creamos DataLoaders
testloader_all  = DataLoader(dataset=test_set_all, batch_size=BATCH_SIZE, shuffle=True)
trainloader_all = DataLoader(dataset=train_set_all, batch_size=BATCH_SIZE, shuffle=True)


#############------------ model 
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

#heart without Batchnorm

# MSE LOSS
data = (trainloader_heart, testloader_heart)
model = define_train_unet(data, BATCH_SIZE, device, "heart-UNet_v0", experiment=True, bn=False)
     
#heart with Batchnorm

# MSE LOSS
data = (trainloader_heart, testloader_heart)
model = define_train_unet(data, BATCH_SIZE, device, "heart-UNet_v1", experiment=True, bn=True)
     
#Binary Cross-Entropy 2D Error
data = (trainloader_heart, testloader_heart)
model = define_train_unet(data, BATCH_SIZE, device, "heart-UNet_v2", loss="CrossEntropy", experiment=True, bn=True)
     

# Dice LOSS
data = (trainloader_heart, testloader_heart)
model = define_train_unet(data, BATCH_SIZE, device, "heart-UNet_v3", loss="Dice", experiment=True, bn=True)
     
     
     
#Lung without Batchnorm

# MSE LOSS
data = (trainloader_lungs, testloader_lungs)
model = define_train_unet(data, BATCH_SIZE, device, "lungs-UNet_v0", experiment=True, bn=False)
     
     
#Lung with Batchnorm
# MSE LOSS
data = (trainloader_lungs, testloader_lungs)
model = define_train_unet(data, BATCH_SIZE, device, "lungs-UNet_v1", experiment=True, bn=True)
     
'''
# BCE 2D LOSS
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
data = (trainloader_lungs, testloader_lungs)
model = define_train_unet(data, BATCH_SIZE, device, "lungs-UNet_v2", loss="CrossEntropy", experiment=True, bn=True)
     
'''
# Dice LOSS
data = (trainloader_lungs, testloader_lungs)
model = define_train_unet(data, BATCH_SIZE, device, "lungs-UNet_v3", loss="Dice", experiment=True, bn=True)
     
#Heart with Batchnorm
# MSE LOSS
data = (trainloader_clav, testloader_clav)
model = define_train_unet(data, BATCH_SIZE, device, "clavicle-UNet_v0", experiment=True, bn=True)
     
# Binary Cross-Entropy 2D LOSS
data = (trainloader_clav, testloader_clav)
model = define_train_unet(data, BATCH_SIZE, device, "clavicle-UNet_v1", loss="CrossEntropy", experiment=True, bn=True)
     
# Dice LOSS
data = (trainloader_clav, testloader_clav)
model = define_train_unet(data, BATCH_SIZE, device, "clavicle-UNet_v2", loss="Dice", experiment=True, bn=True)
     
#Mutlyclass with Batchnorm
# MSE LOSS
data = (trainloader_all, testloader_all)
model = define_train_unet(data, BATCH_SIZE, device, "all-UNet_v0", experiment=True, bn=True, chan_out=3)
     
# BCE LOSS 2D
data = (trainloader_all, testloader_all)
model = define_train_unet(data, BATCH_SIZE, device, "all-UNet_v1", experiment=True, loss="CrossEntropy", bn=True, chan_out=3)
     
# Dice Loss 2D
data = (trainloader_all, testloader_all)
model = define_train_unet(data, BATCH_SIZE, device, "all-UNet_v2", experiment=True, loss="Dice", bn=True, chan_out=3)
     
# Dice Loss 2D
data = (trainloader_all, testloader_all)
model = define_train_unet(data, BATCH_SIZE, device, "all-UNet_v3", experiment=True, loss="CrossEntropy_all", bn=True, chan_out=3)
     
# Log-Cosh Dice Loss
data = (trainloader_all, testloader_all)
model = define_train_unet(data, BATCH_SIZE, device, "all-UNet_v4", experiment=True, loss="log_cosh", bn=True, chan_out=3)
     '''



