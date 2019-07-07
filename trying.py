#------------------------------------------------------------------------------#
#                                                                              #
#  Some initial work for Kaggle competition.                                   #
#  Just figuring out a way to read in the images and orgmize them.             #
#                                                                              #
#------------------------------------------------------------------------------# 


#------------------------------------------------------------------------------#
#                                                                              #
#  Loading relevant libraries                                                  #
#                                                                              #
#------------------------------------------------------------------------------# 

import sys

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from PIL import Image

# need to figure out how to use the following packages
import torch
import torch.nn as nn
import torch.utils.data as D
import torch.nn.functional as F

import torchvision
from torchvision import transforms as T


# this changes the working directory
import os 
os.chdir('C:/Users/dylan/Documents/Recursion-Cellular-Image-Classification')

import tensorflow as tf
from skimage.io import imread
import cv2

#!git clone https://github.com/recursionpharma/rxrx1-utils
#sys.path.append('rxrx1-utils')

import rxrx.io as rio

# this changes the working directory
import os 
#os.chdir('C:/Users/dylan/Documents/Recursion-Cellular-Image-Classification')



#------------------------------------------------------------------------------#
#                                                                              #
#  Trying to read in the data.                                                 #
#                                                                              #
#------------------------------------------------------------------------------# 

# reading the training meta data csv and adding well type indicator
meta_train = pd.read_csv('train.csv')
meta_train['well_type'] = 'non-control' 
print(meta_train.shape)

meta_train_controls = pd.read_csv('train_controls.csv')
print(meta_train_controls.shape)

# combining the meta data (controls are first)
meta_comb = pd.concat([meta_train_controls, meta_train], ignore_index = True)
print(meta_comb.shape)
 


# the following tranforms into array of arrays that are ided by column
# i.e. puts each row of the csv as an elements of an array 
# can then call by index and then .dtype
records = meta_train.to_records(index = False)
experiment1 = records[1].experiment
plate1 = records[1].plate
well1 = records[1].well


#-----------------------------------------------------------------------------#
#                                                                             #
#  a function to specify the file path of the images                          #
#                                                                             #
#-----------------------------------------------------------------------------#

def img_path(meta_data, main_folder, index, site, channel):
    
    # well will be specified from the train.csv
    # channel can be (one of 1,2,3,4,5,6)
    # site can be 1 or 2
    
    experiment = meta_data[index].experiment
    plate = meta_data[index].plate
    well = meta_data[index].well
    
    img_path = main_folder + "/" + str(experiment) + '/Plate' + str(plate)
    
    img_name  = '/' + str(well) + '_s' +str(site) + '_w'+ str(channel) + '.png'
    
    return img_path + img_name
 
# we can now open any image by spcifying a few numbers
meta_records = meta_comb.to_records(index = False)
img = Image.open(img_path(meta_records, 'train', 1, 1, 1))
img

# next, I want to make an rgb image of the 6 layers
 

#------------------------------------------------------------------------------#
#                                                                              #
#  Below, creating RGB image generating function(s)                            #
#                                                                              #
#------------------------------------------------------------------------------# 

## Defining some useful values
DEFAULT_CHANNELS = (1, 2, 3, 4, 5, 6)
RGB_MAP = {
    1: {
        'rgb': np.array([19, 0, 249]),
        'range': [0, 51]
    },
    2: {
        'rgb': np.array([42, 255, 31]),
        'range': [0, 107]
    },
    3: {
        'rgb': np.array([255, 0, 25]),
        'range': [0, 64]
    },
    4: {
        'rgb': np.array([45, 255, 252]),
        'range': [0, 191]
    },
    5: {
        'rgb': np.array([250, 0, 253]),
        'range': [0, 89]
    },
    6: {
        'rgb': np.array([254, 255, 40]),
        'range': [0, 191]
    }
}

## Here, we use function to convert tensor to rgb tensor (3 channel)
def convert_tensor_to_rgb(t, channels=DEFAULT_CHANNELS, vmax=255, rgb_map=RGB_MAP):
    
    colored_channels = []
    
    for i, channel in enumerate(channels):
        x = (t[:, :, i] / vmax) / \
            ((rgb_map[channel]['range'][1] - rgb_map[channel]['range'][0]) / 255) + \
            rgb_map[channel]['range'][0] / 255
        x = np.where(x > 1., 1., x)
        x_rgb = np.array(
            np.outer(x, rgb_map[channel]['rgb']).reshape(512, 512, 3),
            dtype=int)
        colored_channels.append(x_rgb)
    im = np.array(np.array(colored_channels).sum(axis=0), dtype=int)
    im = np.where(im > 255, 255, im)
    return im

## Here, we create the rgb image
def rgb_img(record, main_folder, index, site):
      
    tensor_img = np.ndarray(shape=(512, 512, 6))
    
    for i in range(0,6) :
        
        tensor_img[:,:,i] = Image.open(img_path(record, 
                                            main_folder, 
                                            index, 
                                            site, 
                                            i+1))
    
    rgb_img = convert_tensor_to_rgb(tensor_img)
    
    return(rgb_img, tensor_img)
    
## Running above function
im1_rgb, im1_tensor = rgb_img(meta_records,main_folder = 'train',  index = 10, site = 1)

## Saving image
cv2.imwrite("test3.png", im1)

test = Image.open(img_path(meta_records, "train", 17, 1, 1))
T.ToTensor()(test)
