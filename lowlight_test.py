##################################
## In this code, you can test the model you obtained. 
## Please note that there are specific sections where you need to set paths 
## to files based on your system configuration.
##################################
import torch
import torchvision
import torch.optim
import os
import time
import model
import numpy as np
from PIL import Image
import glob
import time  # noqa: F811

def lowlight(image_path):
	os.environ['CUDA_VISIBLE_DEVICES']='0'
	scale_factor = 1 ## Changed
	# scale_factor = 12
	data_lowlight = Image.open(image_path)

	data_lowlight = (np.asarray(data_lowlight)/255.0)


	data_lowlight = torch.from_numpy(data_lowlight).float()

	h=(data_lowlight.shape[0]//scale_factor)*scale_factor
	w=(data_lowlight.shape[1]//scale_factor)*scale_factor
	data_lowlight = data_lowlight[0:h,0:w,:]
	data_lowlight = data_lowlight.permute(2,0,1)
	data_lowlight = data_lowlight.cuda().unsqueeze(0)

	DCE_net = model.enhance_net_nopool(scale_factor).cuda()
	# please input your model path (for example: ./model_parameter/final/Epoch99.pth)
	DCE_net.load_state_dict(torch.load('model-path/model-name.pth'))
	start = time.time()
	enhanced_image,params_maps = DCE_net(data_lowlight)

	end_time = (time.time() - start)

	print(end_time)
	# set your destination path to save your results
	image_path = image_path.replace('test_metric','outputs/test_metric')

	result_path = image_path
	if not os.path.exists(image_path.replace('/'+image_path.split("/")[-1],'')):
		os.makedirs(image_path.replace('/'+image_path.split("/")[-1],''))
	# import pdb;pdb.set_trace()
	torchvision.utils.save_image(enhanced_image, result_path)
	return end_time

if __name__ == '__main__':

	with torch.no_grad():
		# please input the path of test data
		filePath = 'data/test_metric/'	
		file_list = os.listdir(filePath)
		sum_time = 0
		for file_name in file_list:
			test_list = glob.glob(filePath+file_name+"/*") 
			for image in test_list:

				print(image)
				sum_time = sum_time + lowlight(image)

		print(sum_time)

