import os
import time
import copy
from collections import defaultdict
import torch
import shutil
import pandas as pd
from skimage import io, transform
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, utils
from torch import nn
import albumentations as A
from albumentations.pytorch import ToTensor
#import tqdm as tqdm
from tqdm import tqdm as tqdm

from albumentations import (HorizontalFlip, ShiftScaleRotate, Normalize, Resize, Compose, GaussNoise)
import cv2

from torch.autograd import Variable
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, Dropout
from torch.optim import Adam, SGD
import torch.nn.functional as F
from PIL import Image
import zipfile
import argparse

import random

from core.unet import UNet
from core.loss import DiceBCELoss, DiceLoss, IoU

my_parser = argparse.ArgumentParser()

my_parser.add_argument('-e','--epoch', action='store', type=int, required=True, help="Number of epochs you want to train your model")
my_parser.add_argument('-p','--path', action='store', type=str, required=True, help="Path to Train dataset")
my_parser.add_argument('-o','--out', action='store', type=str, required=True, help="Path to the model weights")
args = my_parser.parse_args()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

TRAIN_PATH = args.path

#Albumentation
def get_train_transform():
   return A.Compose(
       [
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        A.HorizontalFlip(p=0.25),
        A.VerticalFlip(p=0.25),
        ToTensor()
        ])

#Dataset Loader
class LoadDataSet(Dataset):
        def __init__(self,path, transform=None):
            self.path = path
            self.folders = os.listdir(path)
            self.transform = transform
        
        def __len__(self):
            return len(self.folders)
              
        
        def __getitem__(self,idx):
            image_folder    = os.path.join(self.path,self.folders[idx],'images/')
            mask_folder     = os.path.join(self.path,self.folders[idx],'masks/')
            image_path      = os.path.join(image_folder,os.listdir(image_folder)[0])
            
            img             = io.imread(image_path)[:,:,:3].astype('float32')
            img             = transform.resize(img,(128,128))
            
            mask            = self.get_mask(mask_folder, 128, 128 ).astype('float32')

            augmented       = self.transform(image=img, mask=mask)
            img             = augmented['image']
            mask            = augmented['mask']
            mask            = mask[0].permute(2, 0, 1)
            return (img,mask) 


        def get_mask(self,mask_folder,IMG_HEIGHT, IMG_WIDTH):
            mask = np.zeros((IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)
            for mask_ in os.listdir(mask_folder):
                    mask_ = io.imread(os.path.join(mask_folder,mask_))
                    mask_ = transform.resize(mask_, (IMG_HEIGHT, IMG_WIDTH))
                    mask_ = np.expand_dims(mask_,axis=-1)
                    mask = np.maximum(mask, mask_)
              
            return mask

train_dataset = LoadDataSet(TRAIN_PATH, transform=get_train_transform())

## ie, 75% of images in train and left 25% of data in valid
split_ratio             = 0.25
train_size              = int(np.round(train_dataset.__len__()*(1 - split_ratio),0))
valid_size              = int(np.round(train_dataset.__len__()*split_ratio,0))

train_data, valid_data  = random_split(train_dataset, [train_size, valid_size])
train_loader            = DataLoader(dataset=train_data, batch_size=10, shuffle=True)
val_loader              = DataLoader(dataset=valid_data, batch_size=10)

print("Length of train and valid datas: {}, {}".format(len(train_data), len(valid_data)))

## Load Model
model = UNet(3,1).cuda()
optimizer = torch.optim.Adam(model.parameters(),lr = 1e-3)

## Save and Load best model weights.
def save_ckp(state, is_best, checkpoint_path, best_model_path):
    """
    state: checkpoint we want to save
    is_best: is this the best checkpoint; min validation loss
    checkpoint_path: path to save checkpoint
    best_model_path: path to save best model
    """
    f_path = checkpoint_path
    # save checkpoint data to the path given, checkpoint_path
    torch.save(state, f_path)
    # if it is a best model, min validation loss
    if is_best:
        best_fpath = best_model_path
        # copy that checkpoint file to best path given, best_model_path
        shutil.copyfile(f_path, best_fpath)

def load_ckp(checkpoint_fpath, model, optimizer):
    """
    checkpoint_path: path to save checkpoint
    model: model that we want to load checkpoint parameters into       
    optimizer: optimizer we defined in previous training
    """
    # load check point
    checkpoint = torch.load(checkpoint_fpath)
    # initialize state_dict from checkpoint to model
    model.load_state_dict(checkpoint['state_dict'])
    # initialize optimizer from checkpoint to optimizer
    optimizer.load_state_dict(checkpoint['optimizer'])
    # initialize valid_loss_min from checkpoint to valid_loss_min
    valid_loss_min = checkpoint['valid_loss_min']
    # return model, optimizer, epoch value, min validation loss 
    return model, optimizer, checkpoint['epoch'], valid_loss_min.item()

### TRAIN

#from engine import evaluate
criterion       = DiceLoss()
accuracy_metric = IoU()
num_epochs      = args.epoch
valid_loss_min  = np.Inf

model_path      = args.out
checkpoint_path = model_path+'chkpoint_'
best_model_path = model_path+'bestmodel.pt'

if not os.path.exists(model_path):
    os.makedirs(model_path)

total_train_loss    = []
total_train_score   = []
total_valid_loss    = []
total_valid_score   = []

losses_value = 0
for epoch in range(num_epochs):
  
    train_loss  = []
    train_score = []
    valid_loss  = []
    valid_score = []
    #<-----------Training Loop---------------------------->
    pbar = tqdm(train_loader, desc = 'description')
    for x_train, y_train in pbar:
      x_train = torch.autograd.Variable(x_train).cuda()
      y_train = torch.autograd.Variable(y_train).cuda()
      optimizer.zero_grad()
      output = model(x_train)
      #Loss
      loss = criterion(output, y_train)
      losses_value = loss.item()
      #Score
      score = accuracy_metric(output,y_train)
      loss.backward()
      optimizer.step()
      train_loss.append(losses_value)
      train_score.append(score.item())
      #train_score.append(score)
      pbar.set_description(f"Epoch: {epoch+1}, loss: {losses_value}, IoU: {score}")

    #<---------------Validation Loop---------------------->
    with torch.no_grad():
      for image,mask in val_loader:
        image = torch.autograd.Variable(image).cuda()
        mask = torch.autograd.Variable(mask).cuda()
        output = model(image)
        ## Compute Loss Value.
        loss = criterion(output, mask)
        losses_value = loss.item()
        ## Compute Accuracy Score
        score = accuracy_metric(output,mask)
        valid_loss.append(losses_value)
        valid_score.append(score.item())

    total_train_loss.append(np.mean(train_loss))
    total_train_score.append(np.mean(train_score))
    total_valid_loss.append(np.mean(valid_loss))
    total_valid_score.append(np.mean(valid_score))
    print(f"\n###############Train Loss: {total_train_loss[-1]}, Train IOU: {total_train_score[-1]}###############")
    print(f"###############Valid Loss: {total_valid_loss[-1]}, Valid IOU: {total_valid_score[-1]}###############")

    #Save best model Checkpoint
    # create checkpoint variable and add important data
    checkpoint = {
        'epoch': epoch + 1,
        'valid_loss_min': total_valid_loss[-1],
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }
    
    # save checkpoint
    save_ckp(checkpoint, False, checkpoint_path, best_model_path)
    
    ## TODO: save the model if validation loss has decreased
    if total_valid_loss[-1] <= valid_loss_min:
        print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(valid_loss_min,total_valid_loss[-1]))
        # save checkpoint as best model
        save_ckp(checkpoint, True, checkpoint_path, best_model_path)
        valid_loss_min = total_valid_loss[-1]

