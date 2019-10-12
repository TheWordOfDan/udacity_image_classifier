'''
Filename: 		predict.py
Author:			Daniel Jimenez
Date created:	20190921
Last Modified:	20191007
'''

import utils
import torch
import torchvision
from torchvision import transforms, datasets
import numpy as np
import json
import argparse
import PIL
from PIL import Image



parser = argparse.ArgumentParser()
parser.add_argument('path_to_image', action = 'store')
parser.add_argument('--gpu', action = 'store_true')
parser.add_argument(
	'--json_file', 
	action = 'store', 
	default = './cat_to_name.json')
parser.add_argument('--top_k', action = 'store', type = int, default = 5)
parser.add_argument('--checkpoint', action = 'store', default = './checkpoint.pth')


#---GPU?---
gpuRequested = parser.parse_args().gpu
device = 'cpu'
if gpuRequested:
	if torch.cuda.is_available():
		device = 'cuda'
		print('\nGPU detected!')
	else:
		print('\nNo GPU detected! Using CPU.')



#---Load the image---
# (convert it to a torch tensor)
path = parser.parse_args().path_to_image
pil_image = Image.open(path)
transforms = transforms.Compose([
	transforms.Resize(255),
	transforms.CenterCrop(224),
	transforms.ToTensor(),
	transforms.Normalize([0.485,0.456,0.406],
						 [0.229,0.225,0.225])])
image = transforms(pil_image)
image = image.unsqueeze(0)
image = image.float()



#---Load the checkpoint---
model = utils.loadNN(parser.parse_args().checkpoint)


#---use gpu/or cpu for making the prediction---
model.to(device)
with torch.no_grad():
	if device == 'cuda':
		output = model.forward(image.cuda())
	if device == 'cpu':
		output = model.forward(image)


#---Predict!---
ps = torch.exp(output)
ps = torch.nn.functional.softmax(output, dim=1)
top_p, top_class = ps.topk(parser.parse_args().top_k, dim=1)



#---JSON file to match top_class to correct name
with open(parser.parse_args().json_file, 'r') as f:
	cat_to_name = json.load(f)



#---match top_class to correct name, print results
idx_to_label = {}
for key in model.class_to_idx:
	idx_to_label[model.class_to_idx[key]] = cat_to_name[key]
top_p = top_p.cpu()
top_class = top_class.cpu()
percentages = [i for i in np.array(top_p)[0]]
predictions = [idx_to_label[i] for i in np.array(top_class)[0]]

print('\n --- Results: ---')
for i in range(parser.parse_args().top_k):
	print(f'{i+1}) {percentages[i]*100:.2f}%	{predictions[i]}')
print('\n')

