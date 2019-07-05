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
 




