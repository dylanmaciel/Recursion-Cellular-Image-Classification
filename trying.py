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

# so we need to use the cs files to specify the path when getting images
# keep track of well_type

# finding the image path

for i in range(1, meta_train.shape[1]):
    experiment = meta_train[i].experiment
    well = meta_train[i].well
    plate = meta_train[i].plate


