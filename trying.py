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

# reading the training meta data csv
meta_train = pd.read_csv('train.csv')
print(meta_train.shape)

meta_train_controls = pd.read_csv('train_controls.csv')
print(meta_train_controls.shape)

#  the following imports one of the specified files 
im = Image.open('./B02/B02_s1_w1.png')

# so we need to use the csv files to specify the path for getting images



