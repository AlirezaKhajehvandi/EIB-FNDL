##################################
## In this code, you can train the network and obtain your trained model. 
## Please note that there are specific sections where you need to set paths 
## to files based on your system configuration.
## Note: We used training data from the ZeroDCE++ `train_data` dataset
## ZeroDCE++ Link: "https://github.com/Li-Chongyi/Zero-DCE_extension"

## This code is adapted from "https://github.com/Li-Chongyi/Zero-DCE_extension" and we changed it for improvement.
## License: Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)
##################################


import torch
import torch.optim
import os
import argparse
import dataloader
import model
import Myloss


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)





def train(config):
	os.environ['CUDA_VISIBLE_DEVICES']='0'
	scale_factor = config.scale_factor
	DCE_net = model.enhance_net_nopool(scale_factor).cuda()

	# DCE_net.apply(weights_init)
	if config.load_pretrain == True:  # noqa: E712
	    DCE_net.load_state_dict(torch.load(config.pretrain_dir))

	train_dataset = dataloader.lowlight_loader(config.lowlight_images_path)	
	train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.train_batch_size, shuffle=True, num_workers=config.num_workers, pin_memory=True)

	L_color = Myloss.L_color()
	L_spa = Myloss.L_spa()
	L_exp = Myloss.L_exp(16)
	L_TV = Myloss.L_TV()
	L_bright = Myloss.L_brightness()
	L_color_ratio = Myloss.L_color_ratio()

	optimizer = torch.optim.Adam(DCE_net.parameters(), lr=config.lr, weight_decay=config.weight_decay)
	
	DCE_net.train()

	for epoch in range(config.num_epochs):
		for iteration, img_lowlight in enumerate(train_loader):

			img_lowlight = img_lowlight.cuda()

			E = 0.6

			enhanced_image,A  = DCE_net(img_lowlight)

			# ZeroDCE++ Loss functions
			Loss_TV = 1600*L_TV(A)	
			loss_spa = torch.mean(L_spa(enhanced_image, img_lowlight))
			loss_col = 5*torch.mean(L_color(enhanced_image)) # org_coef = 5

			# The proposed loss functions
			loss_col_ratio = 5*torch.mean(L_color_ratio(img_lowlight, enhanced_image))  
			loss_exp = 10*torch.mean(L_exp(img_lowlight, enhanced_image,E))
			loss_bright = torch.mean(L_bright(enhanced_image))


			# total loss function
			loss =  Loss_TV + loss_spa + loss_col + loss_col_ratio + loss_bright + loss_exp
			
			optimizer.zero_grad()
			loss.backward()
			torch.nn.utils.clip_grad_norm(DCE_net.parameters(),config.grad_clip_norm)
			optimizer.step()

			if ((iteration+1) % config.display_iter) == 0:
				print(f"***************** Epoch {epoch + 1} | Iteration : {iteration + 1} **********************")				
				print(f"loss bright: {loss_bright}")
				print(f"loss TV: {Loss_TV}")
				print(f"loss spa: {loss_spa}")
				print(f"loss color: {loss_col}")
				print(f"loss exp: {loss_exp}")
				print(f"loss col ratio: {loss_col_ratio}")
				print(f"total loss: {loss}")				

			if ((iteration+1) % config.snapshot_iter) == 0:
				
				torch.save(DCE_net.state_dict(), config.snapshots_folder + "Epoch" + str(epoch) + '.pth') 		




if __name__ == "__main__":

	parser = argparse.ArgumentParser()

	# Input Parameters
	parser.add_argument('--lowlight_images_path', type=str, default="data/train_data/")
	parser.add_argument('--lr', type=float, default=0.0001)
	parser.add_argument('--weight_decay', type=float, default=0.0001)
	parser.add_argument('--grad_clip_norm', type=float, default=0.1)
	parser.add_argument('--num_epochs', type=int, default=11)
	parser.add_argument('--train_batch_size', type=int, default=8)
	parser.add_argument('--val_batch_size', type=int, default=8)
	parser.add_argument('--num_workers', type=int, default=4)
	parser.add_argument('--display_iter', type=int, default=10)
	parser.add_argument('--snapshot_iter', type=int, default=10)
	parser.add_argument('--scale_factor', type=int, default=1)
	parser.add_argument('--snapshots_folder', type=str, default="snapshot_epochs/")
	parser.add_argument('--load_pretrain', type=bool, default= False)
	parser.add_argument('--pretrain_dir', type=str, default= "snapshots_Zero_DCE++/Epoch99.pth")

	config = parser.parse_args()

	if not os.path.exists(config.snapshots_folder):
		os.mkdir(config.snapshots_folder)


	train(config)








	
