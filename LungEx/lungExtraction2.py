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

from tqdm import tqdm
from comet_ml import Experiment


#Global Variables 
IMG_SIZE = 224
SIZE = 224

def num2fixstr(x,d):
    # example num2fixstr(2,5) returns '00002'
    # example num2fixstr(19,3) returns '019'
    st = '%0*d' % (d,x)
    return st
        

#classes for the UNet
class Unet(nn.Module):

    # Implementation of U-Net inherating from nn.Module
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
    def conv_layer(self, n, l_n, batchnorm=False):
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
    # Source: https://github.com/hiyouga/Image-Segmentation-PyTorch/blob/master/loss_func.py
    def __init__(self):
        super(BCELoss2d, self).__init__()
        self.bce_loss = nn.BCELoss()
        
    def forward(self, predict, target):
        predict = predict.view(-1)
        target = target.view(-1)
        return self.bce_loss(predict, target)

class BCELoss3d(nn.Module):
    # simialuar to BCELoss2d but for seperate classes
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
    #normal dice loss for one class
    # also works for multible classes but do not seperate them
    # Source: https://github.com/pytorch/pytorch/issues/1249
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
    # training one epoch
    # inlcudes the logging of the loss
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
        # using COMET for logging the loss
        experiment.log_metric("train_loss", epoch_loss / len(iterator), step=e)
        
    return epoch_loss/len(iterator)


def eval_epoch(model, iterator, criterion, device, experiment, e):

    #initialize every epoch
    epoch_loss = 0

    #deactivating dropout layers
    model.eval()

    #deactivates autograd
    with torch.no_grad():
        for batch in tqdm(iterator):
            
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
        # using COMET for logging the loss
        experiment.log_metric("val_loss", epoch_loss / len(iterator), step=e)
            
    return epoch_loss / len(iterator)


def train_epochs(model, data, criterion, optimizer, device, epochs, experiment, name, early=10):
    # trains one model n epochs
    # also uses COMET to logg the loss
    best_val = float("Inf")

    data_train, data_val = data
    early_stop = 0

    for epoch in tqdm(range(epochs), desc= "Epochs"):
    
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

def load_images(fpath, img_names, nl=IMG_SIZE, nc=IMG_SIZE):
    # load the images and save them in dict
    images = {}
    masks = {}
    for name in img_names:
        st = fpath + os.sep + name
        # get the images ending with .tif
        # https://forum.image.sc/t/how-to-open-the-images-from-jsrt-database-in-imagej/11317/7
        print("Name: ", name)
        X = imread(st)
        X = cv2.resize(X, (nl, nc)) / 255
        # plt.imshow(X, cmap="gray")
        X = X.astype(np.double)
        images[name.split('.')[0]] = X
        
        #loading the masks
        path_mask = st.replace("Images", "Masks")
        mask = imread(path_mask)
        mask = cv2.resize(mask, (nl, nc)) / 255
        masks[name.split('.')[0]] = mask
        
    return images, masks





##########--------- creating Pytorch Dataset
def trans(data):
    # Helper function
    return torch.unsqueeze(torch.from_numpy(data), dim=0).float()

def trans_(data):
    # Helper function
    return torch.from_numpy(data).float()
        

class JSRTDataset(Dataset):

    def __init__(self, files, im_dict = 'Images'):
        super().__init__()
        images, masks = load_images(im_dict, files)
        self.data = []
        self.add_data(images, masks)

        
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




######---- Dataloader lung
BATCH_SIZE = 2

# Test y train dataset
im_dict = "Images"
files  = dirfiles(im_dict, "*.tif")

test_files = [files[-1]]
train_files = files[:-1]

print("test Files: ", test_files, type(test_files))

test_set_lungs  = JSRTDataset(test_files, im_dict)
train_set_lungs = JSRTDataset(train_files, im_dict)

# Creamos DataLoaders
testloader_lungs  = DataLoader(dataset=test_set_lungs, batch_size=BATCH_SIZE, shuffle=True)
trainloader_lungs = DataLoader(dataset=train_set_lungs, batch_size=BATCH_SIZE, shuffle=True)




# BCE 2D LOSS
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
data = (trainloader_lungs, testloader_lungs)
model = define_train_unet(data, BATCH_SIZE, device, "lungs-UNet_v2", loss="CrossEntropy", experiment=False, bn=True)
     




