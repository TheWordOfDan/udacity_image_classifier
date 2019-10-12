'''
Filename:		train.py
Author: 		Daniel Jimenez
Date created:	20190920 (original), 20191001 (this branch)
Last modified: 	20191007
'''

import utils
import torch
from torch import nn
from torch import optim

import matplotlib.pyplot as plt

import torchvision
from torchvision import datasets, transforms, models

import argparse
parser = argparse.ArgumentParser()
parser.add_argument(
	'--data_directory', 
	action = 'store', 
	default = './flowers')
parser.add_argument(
	'--gpu', 
	action = 'store_true')
parser.add_argument(
	'--learning_rate', 
	action = 'store', 
	default = 0.004)
parser.add_argument(
	'--hidden_units',
	action = 'store',
	default = 2048)
parser.add_argument(
	'--epochs',
	action = 'store',
	default = 5)
parser.add_argument(
	'--arch',
	action = 'store',
	type = str,
	default = 'vgg13')



#---Set data directories---
data_dir = parser.parse_args().data_directory
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'



#---GPU?---
gpuRequested = parser.parse_args().gpu
device = 'cpu'
if gpuRequested:
	if torch.cuda.is_available():
		device = 'cuda'
		print('\nGPU detected!')
	else:
		print('\nNo GPU detected! Using CPU.')



#---Check learning rate---
learning_rate = parser.parse_args().learning_rate
try:
	learning_rate = float(learning_rate)
except:
	print(f'Error - invalid input for learning_rate - ({learning_rate})'
		f' - must be a float (default: 0.005)')
	exit()



#---Check epochs---
epochs = parser.parse_args().epochs
try:
	epochs = int(epochs)
except:
	print(f'Error - invalid input for epochs - ({epochs})'
		f' - must be an integer (default: 1)')
	exit()



#---Get pretrained architecture---
pretrained_input_sizes = utils.getPretrainedInputSizes()
recognized = [key for key in pretrained_input_sizes.keys()]
arch = parser.parse_args().arch
model = utils.getPretrained(arch)
if model == None:
	print(f'Error: unrecognized architecture - "{arch}".'
		f'\nPossible options: {str([i for i in recognized])[1:-1]}')
	exit()



#---Set classifier layers--- 
hidden_units = parser.parse_args().hidden_units
try:
	hidden_units = int(hidden_units)
except:
	print(f'Error - invalid input for hidden_units - ("{hidden_units}")'
		f' - must be an integer. (default: 2048)')
	exit()

nnLayers = {'input': pretrained_input_sizes[arch],  
			'hidden': [hidden_units, 1024],
			'output': 102,
			'pretrained': arch}



#---Create classifier---
classifier = utils.createNN(nnLayers)



#---Attach classifier to pretrained, set loss, set optimizer---
model = utils.attachClassifier(model, arch, classifier)
criterion = nn.NLLLoss()
optimizer = optim.Adam(classifier.parameters(), lr = learning_rate)



#---Final confimation---
print(f'\nConfirm final settings: ',
	f'\n  Architecture: 	{arch}',
	f'\n  Hidden units: 	{hidden_units}',
	f'\n  Learning rate: 	{learning_rate}',
	f'\n  Epochs:		{epochs}',
	f'\n  Process with: 	{device}')
while True:
	userInput = input('\nProceed? [y/n] ')
	if userInput.upper() not in ['Y', 'N']:
		print('Please enter "y" or "n"!')
		continue
	else:
		if userInput.upper() == 'N':
			print('\nAborted!\n')
			exit()
		else:
			break



#---Get dataloaders---
trainloader, testloader, validloader, class_to_idx = utils.getDataLoaders(
	train_dir,
	test_dir,
	valid_dir)



#---TRAINING MODE---
print('\nENTERING TRAINING MODE...')
model.to(device)
model.train()
running_loss = 0
validationFreq = 50
validationCountdown = validationFreq
validation_loss, training_loss = [], []

for epoch in range(epochs):
	print(f'Beginning epoch: {epoch + 1} of {epochs}')
	
	#---TRAINING PASS---
	for inputs, labels in trainloader:
		running_vloss = 0
		vbatch_accuracy = 0

		
		validationCountdown -= 1
		
		
		optimizer.zero_grad()
		inputs, labels = inputs.to(device), labels.to(device)
		log_ps = model(inputs)
		loss = criterion(log_ps, labels)
		loss.backward()
		optimizer.step()
		running_loss += loss.item()

		
		#---VALIDATION PASS---
		# Occurs every nth training batch, where n is the value
		# stored in validationFreq
		if validationCountdown > 0:
			continue


		with torch.no_grad():
			validationCountdown = validationFreq
			model.eval()
			for vinputs, vlabels in validloader:
				vinputs, vlabels = vinputs.to(device), vlabels.to(device)
				vlog_ps = model(vinputs)
				vloss = criterion(vlog_ps, vlabels)
				running_vloss += vloss.item()
				vps = torch.exp(vlog_ps)
				top_p, top_class = vps.topk(1, dim = 1)
				equals = (top_class == vlabels.view(*top_class.shape))
				accuracy = torch.mean(equals.type(torch.FloatTensor))
				vbatch_accuracy += accuracy.item()
			

			# This is for the running output
			output = []
			output.append(running_loss/validationFreq)
			output.append(running_vloss/len(validloader))
			output.append(vbatch_accuracy/len(validloader)*100)

			
			# These are for plotting the results at the end
			training_loss.append(output[0])
			validation_loss.append(output[1])
			
			
			print(f'Running loss: {output[0]:.3f},',
				f'Validation loss: {output[1]:.3f},',
				f'Validation accuracy: {output[2]:.3f}%')
			running_loss = 0
		
		
		model.train()



#---TESTING PHASE---
print('\nTesting Phase:')
model.eval()
test_accuracy = 0
with torch.no_grad():
	for images, labels in testloader:
		images, labels = images.to(device), labels.to(device)
		ps = torch.exp(model(images))
		(top_p, top_class) = ps.topk(1, dim=1)
		equals = (top_class == labels.view(*top_class.shape))
		accuracy = torch.mean(equals.type(torch.FloatTensor))
		test_accuracy += accuracy.item()
	print(f'Accuracy on test dataset: {(test_accuracy / len(testloader))*100:.3f}%\n')



#---Save---
checkpoint = nnLayers
checkpoint['state_dict'] = classifier.state_dict()
checkpoint['class_to_idx'] = class_to_idx
checkpoint['epochs'] = epochs
torch.save(checkpoint, 'checkpoint.pth')



#---Display results---
plt.plot(validation_loss, label = 'Validation Loss')
plt.plot(training_loss, label = 'Training Loss')
plt.legend(frameon = False)
plt.show()



#---Quit---
print('\nTraining complete. Classifier saved to checkpoint.pth\n')
