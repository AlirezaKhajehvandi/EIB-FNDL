import os
import sys

import torch
import torch.utils.data as data

import numpy as np
from PIL import Image
import glob
import random


random.seed(1143)

# input map (formula)
def input_map(x):
    scaled_x = ((1.5 + (1 * x - 0.5) * np.pi))
    
    # Calculate the modified sine value
    sine_value = np.cos(scaled_x)
    
    # Scale and shift the sine value to map [-1, 1] to [0, 1]
    y = 0.4 * (sine_value + 1) + 0.1
    
    return y


def populate_train_list(lowlight_images_path):

	image_list_lowlight = glob.glob(lowlight_images_path + "*.jpg")

	train_list = image_list_lowlight

	random.shuffle(train_list)

	return train_list

	

class lowlight_loader(data.Dataset):

	def __init__(self, lowlight_images_path):

		self.train_list = populate_train_list(lowlight_images_path) 
		self.size = 512

		self.data_list = self.train_list
		print("Total training examples:", len(self.train_list))


		

	def __getitem__(self, index):

		data_lowlight_path = self.data_list[index]
		data_lowlight = Image.open(data_lowlight_path)
		# data_lowlight = data_lowlight.resize((self.size,self.size), Image.ANTIALIAS) # Image.ANTIALIAS is no longer support
		data_lowlight = data_lowlight.resize((self.size,self.size), Image.Resampling.LANCZOS) 
		data_lowlight = ((np.asarray(data_lowlight))/255.0) 
		data_lowlight = 1 - data_lowlight # simple inverse
		
		data_lowlight = torch.from_numpy(data_lowlight).float()

		return data_lowlight.permute(2,0,1)

	def __len__(self):
		return len(self.data_list)

