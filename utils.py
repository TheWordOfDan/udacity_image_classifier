'''
Filename:		utils.py
Author:			Daniel Jimenez
Date created:	20190920
Last modified:	20191007
'''

import torch
from torch import nn

import torchvision
from torchvision import datasets, transforms, models



def createNN(layers_):
	'''
		layers_['hidden'] is a list; to deal with the unknown length
		the requried lines of code to account for each item is added
		onto a string and then evaluated at runtime
		
		The relevant user input has been casted into an int or else rejected
		beforehand, so there's no way for the user to force the interpreter
		to execute arbitrary code

		Given: 
		layers_ - a dictionary containing a nn architecture 

		Returns:
		an untrained nn with the given architecture 
	'''
	params = "nn.Sequential(nn.Linear(layers_['input']"
	for i in range(len(layers_['hidden'])):
		params += f", layers_['hidden'][{i}])"
		params += ", nn.ReLU(), nn.Dropout(p=0.1)"
		params += f", nn.Linear(layers_['hidden'][{i}]"
	params += ", layers_['output']), nn.LogSoftmax(dim = 1))"
	classifier = eval(params)


	return classifier



def loadNN(filename_ = './checkpoint.pth'):
	'''
		Called from predict.py; loads the architecture and state_dict data 
		from a file, constructs a nn with it, and returns it to predict.py.

		Given:
		filename_	a string with the location of a 'checkpoint.pth' file

		Returns:
		A nn constructed from the checkpoint stored in the given file
	'''
	filename = filename_
	checkpoint = torch.load(filename)
	classifier = createNN(checkpoint)
	classifier.load_state_dict(checkpoint['state_dict'])
	arch = checkpoint['pretrained']
	model = getPretrained(arch)	
	model.class_to_idx = checkpoint['class_to_idx']
	model = attachClassifier(model, arch, classifier)

	
	return model



def getPretrainedInputSizes():
	'''
		The one stop for which pretrained models are recognized.

		Given:
		- (none)

		Returns:
		- a dictionary with
			- k: the name of the pretrained model
			- v: the number of inputs (needed to attach a classifier)
	'''
	pretrained_input_sizes = {'vgg13':4096,
							  'resnet50':2048,
							  'densenet121':1024}


	return pretrained_input_sizes



def getPretrained(arch_):
	'''
		Given the string containing the name of the pretrained
		model, returns the correct pretrained model.
		
		Given:
		- arch_		a string with the name of the pretrained model
		
		Returns:
		- The pretrained model, with features set in place.

	'''
	recognized = getPretrainedInputSizes().keys()
	if arch_ not in recognized:
		return None

	
	command = f'models.{arch_}(pretrained = True)'
	model = eval(command)


	for param in model.parameters():
		param.requires_grad = False


	return model	



def attachClassifier(model_, arch_, classifier_):
	'''
		Takes the user defined classifier and attaches it
		to the chosen prebuilt model.

		Given:
		- model_		The pretrained model
		- arch_			a string with the name of the 
						classifier architecture
		- classifier_	The classifier itself

		Returns:
		- the model, ready to train or predict
	'''
	if arch_ == 'vgg13':
		model_.classifier[6] = classifier_
	if arch_ == 'resnet50':
		model_.fc = classifier_	
	if arch_ == 'densenet121':
		model_.classifier = classifier_
	
	
	return model_



def getDataLoaders(trainDir_, testDir_, validDir_):
	'''
		Called from train.py, defines the transforms for the dataloaders,
		creates them, then gives them back to train.py.

		Given:
		- trainDir_	a string with the location of the training data
		- testDir_	a string with the location of the test data
		- validDir_ a string with the location of the validation data

		Returns:
		- three dataloaders with the training data, test data, and validation data
	'''
	train_transforms = transforms.Compose([
		transforms.RandomRotation(30),
		transforms.RandomResizedCrop(224),
		transforms.RandomHorizontalFlip(),
		transforms.ToTensor(),
		transforms.Normalize([0.485,0.456,0.406],
							 [0.229,0.225,0.225])
		])	


	test_transforms = transforms.Compose([
		transforms.Resize(255),
		transforms.CenterCrop(224),
		transforms.ToTensor(),
		transforms.Normalize([0.485,0.456,0.406],
							 [0.229,0.225,0.225])
		])

	
	valid_transforms = transforms.Compose([
		transforms.Resize(255),
		transforms.CenterCrop(224),
		transforms.ToTensor(),
		transforms.Normalize([0.485,0.456,0.406],
							 [0.229,0.225,0.225])
		])	
	

	train_data = datasets.ImageFolder(trainDir_, transform = train_transforms)
	test_data = datasets.ImageFolder(testDir_, transform = test_transforms)
	valid_data = datasets.ImageFolder(validDir_, transform = valid_transforms)

	
	trainloader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)
	testloader = torch.utils.data.DataLoader(test_data, batch_size=32, shuffle=True)
	validloader = torch.utils.data.DataLoader(valid_data, batch_size=32, shuffle=True)


	return (trainloader, testloader, validloader, train_data.class_to_idx)



if __name__ == '__main__':
	'''
	Testing code goes here
	'''
	classifier = loadNN(None)
	print(classifier)
