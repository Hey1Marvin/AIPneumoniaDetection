#pip install torch torchvision tqdm


import pydicom
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import functional as F2
from torch.nn import init
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import numpy as np
import pandas as pd
from PIL import Image



# Setup paths
model_path = 'best_model_atun.pth'
images_dir = 'rsna-pneumonia-dataset/stage_2_train_images'
label_file = 'rsna-pneumonia-dataset/stage_2_train_labels.csv'
target_dir = 'rsna-pneumonia-dataset/images_transformed'


#Model / learning Parameters
num_epochs = 50
learningrate = 0.001
batch_size = 8


class ChestXrayDataset(Dataset):
    def __init__(self, images_dir, masks_dir, transform=None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform
        self.images = sorted(os.listdir(images_dir))
        self.masks = sorted(os.listdir(masks_dir))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.images_dir, self.images[idx])
        mask_path = os.path.join(self.masks_dir, self.images[idx].replace(".", "_mask."))

        # Lade das Röntgenbild (Grayscale)
        image = Image.open(img_path).convert('L')

        # Lade die Maske (Grayscale)
        mask = Image.open(mask_path).convert('L')

        if self.transform:
            image = self.transform(image)
            # Wenn dein Modell 3 Kanäle benötigt, du aber Graustufenbilder hast:
            image = image.repeat(3, 1, 1)  # Wiederhole den Graustufenkanal 3 mal, um 3 Kanäle zu erhalten
            mask = self.transform(mask)


        # Konvertiere Maske zu binärer Maske (Werte 0 oder 1)
        mask = torch.where(mask > 0.5, 1.0, 0.0)

        return image, mask, self.images[idx]



class PneumoniaDataset(Dataset):
    def __init__(self, image_dir, df, transforms=None):
        self.image_dir = image_dir
        self.transforms = transforms
        df = pd.read_csv(df)
        self.labels = df
        self.labels.fillna(-1, inplace=True)
        self.labels['Target'] = self.labels['Target'].astype(int)
        self.image_ids = self.labels['patientId'].unique()

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image_path = os.path.join(self.image_dir, image_id + '.dcm')
        dicom = pydicom.read_file(image_path)
        image = dicom.pixel_array
        if len(image.shape) == 2:  # Grayscale image
            image = np.stack([image] * 3, axis=2)
        image = F2.to_tensor(image).float()
        
        boxes = []
        masks = []
        labels = []
        
        patient_data = self.labels[self.labels['patientId'] == image_id]
        for _, row in patient_data.iterrows():
            if row['Target'] == 1:
                boxes.append([row['x'], row['y'], row['x'] + row['width'], row['y'] + row['height']])
                mask = np.zeros((image.shape[1], image.shape[2]), dtype=np.uint8)
                mask[int(row['y']):int(row['y'] + row['height']), int(row['x']):int(row['x'] + row['width'])] = 1
                masks.append(mask)
                labels.append(1)
            else:
                labels.append(0)
        
        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            masks = torch.zeros((0, image.shape[1], image.shape[2]), dtype=torch.uint8)
            labels = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            masks = torch.as_tensor(masks, dtype=torch.uint8)
            labels = torch.as_tensor(labels, dtype=torch.int64)
        
        target = {
            'boxes': boxes,
            'labels': labels,
            'masks': masks,
        }
        
        if self.transforms:
            image, target = self.transforms(image, target)
        
        return image, image_id


######### Attention U-Net ################
def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)

class conv_block(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(conv_block,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )


    def forward(self,x):
        x = self.conv(x)
        return x

class up_conv(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(up_conv,self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in,ch_out,kernel_size=3,stride=1,padding=1,bias=True),
		    nn.BatchNorm2d(ch_out),
			nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = self.up(x)
        return x

class Recurrent_block(nn.Module):
    def __init__(self,ch_out,t=2):
        super(Recurrent_block,self).__init__()
        self.t = t
        self.ch_out = ch_out
        self.conv = nn.Sequential(
            nn.Conv2d(ch_out,ch_out,kernel_size=3,stride=1,padding=1,bias=True),
		    nn.BatchNorm2d(ch_out),
			nn.ReLU(inplace=True)
        )

    def forward(self,x):
        for i in range(self.t):

            if i==0:
                x1 = self.conv(x)
            
            x1 = self.conv(x+x1)
        return x1
        
class RRCNN_block(nn.Module):
    def __init__(self,ch_in,ch_out,t=2):
        super(RRCNN_block,self).__init__()
        self.RCNN = nn.Sequential(
            Recurrent_block(ch_out,t=t),
            Recurrent_block(ch_out,t=t)
        )
        self.Conv_1x1 = nn.Conv2d(ch_in,ch_out,kernel_size=1,stride=1,padding=0)

    def forward(self,x):
        x = self.Conv_1x1(x)
        x1 = self.RCNN(x)
        return x+x1


class single_conv(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(single_conv,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = self.conv(x)
        return x

class Attention_block(nn.Module):
    def __init__(self,F_g,F_l,F_int):
        super(Attention_block,self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
            )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self,g,x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1+x1)
        psi = self.psi(psi)

        return x*psi

class AttU_Net(nn.Module):
    def __init__(self,img_ch=3,output_ch=1):
        super(AttU_Net,self).__init__()
        
        self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)

        self.Conv1 = conv_block(ch_in=img_ch,ch_out=64)
        self.Conv2 = conv_block(ch_in=64,ch_out=128)
        self.Conv3 = conv_block(ch_in=128,ch_out=256)
        self.Conv4 = conv_block(ch_in=256,ch_out=512)
        self.Conv5 = conv_block(ch_in=512,ch_out=1024)

        self.Up5 = up_conv(ch_in=1024,ch_out=512)
        self.Att5 = Attention_block(F_g=512,F_l=512,F_int=256)
        self.Up_conv5 = conv_block(ch_in=1024, ch_out=512)

        self.Up4 = up_conv(ch_in=512,ch_out=256)
        self.Att4 = Attention_block(F_g=256,F_l=256,F_int=128)
        self.Up_conv4 = conv_block(ch_in=512, ch_out=256)
        
        self.Up3 = up_conv(ch_in=256,ch_out=128)
        self.Att3 = Attention_block(F_g=128,F_l=128,F_int=64)
        self.Up_conv3 = conv_block(ch_in=256, ch_out=128)
        
        self.Up2 = up_conv(ch_in=128,ch_out=64)
        self.Att2 = Attention_block(F_g=64,F_l=64,F_int=32)
        self.Up_conv2 = conv_block(ch_in=128, ch_out=64)

        self.Conv_1x1 = nn.Conv2d(64,output_ch,kernel_size=1,stride=1,padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self,x):
        # encoding path
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)
        
        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        x4 = self.Att5(g=d5,x=x4)
        d5 = torch.cat((x4,d5),dim=1)        
        d5 = self.Up_conv5(d5)
        
        d4 = self.Up4(d5)
        x3 = self.Att4(g=d4,x=x3)
        d4 = torch.cat((x3,d4),dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        x2 = self.Att3(g=d3,x=x2)
        d3 = torch.cat((x2,d3),dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        x1 = self.Att2(g=d2,x=x1)
        d2 = torch.cat((x1,d2),dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)
        
        out = self.sigmoid(d1)

        return out

    
    





def crop_to_lung(image, mask):
    """
    Crops the image and mask to the region containing the lungs (non-zero regions in the mask).
    """
    # Convert mask to binary
    binary_mask = mask > 0.5

    # Get bounding box of the lungs
    nonzero_coords = torch.nonzero(binary_mask)
    if len(nonzero_coords) == 0:
        # Return the original image and mask if no lung is detected
        return image, mask

    min_y, min_x = torch.min(nonzero_coords, dim=0)[0]
    max_y, max_x = torch.max(nonzero_coords, dim=0)[0]

    # Crop both the image and the mask
    cropped_image = image[:, min_y:max_y+1, min_x:max_x+1]
    cropped_mask = mask[:, min_y:max_y+1, min_x:max_x+1]

    return cropped_image, cropped_mask


def transPic(model, device, data_loader, target_dir, transform):
    model.eval()

    with torch.no_grad():
        for images, image_id in tqdm(data_loader):
            #print("Image size 1:",images[0].size())
            transImg = torch.stack([transform(image) for image in images])
            
            transImg = transImg.to(device)
            outputs = model(transImg)

            #print("Image size 2:",images[0].size(), " outputs: ", outputs[0].size())
            # Maske erzeugen und auf Ursprungsgroesse hochsampeln
            up = nn.Upsample(size=images[0].size()[1:], mode='bilinear')
            masks = up(outputs)
            #print("Image size 3:",images[0].size()," Out: " ,masks[0].size())
            # Verarbeite Bilder und Masken
            for image, mask, img_id in zip(images, masks, image_id):
                out = image[0] * mask
                # Angenommen, dein Tensor heißt 'image_tensor'
                #print("Image size 4:",out.size())
                out = out.squeeze().numpy()
                out = (out * 255).astype(np.uint8)
                #
                # png Datei speichern
                im = Image.fromarray(out)
                im.save(target_dir+"/"+img_id+'.jpeg')





if __name__ == '__main__':
    # Transformations for the dataset
    transform = transforms.Compose([
        transforms.Resize((256, 256))
    ])

    transform2 = transforms.Compose([
        transforms.ToTensor()
    ])

    # Load dataset
    dataset = PneumoniaDataset(images_dir, label_file)

   

    
    # Create data loaders
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize the model, loss function, optimizer, and learning rate scheduler
    model = AttU_Net()
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    

    # Move model to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    transPic(model, device, data_loader, target_dir, transform)


