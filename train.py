# Imports here
import torch
from torchvision import datasets, transforms , models
from torch import nn, optim
import torch.nn.functional as F
from collections import OrderedDict
from PIL import Image
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import pros 
import argparse
    #models
    # Command line arguments
parser = argparse.ArgumentParser(description='Image ClassifierTrain model')
parser.add_argument('--arch', type = str, default = 'densenet121', help = 'NN Model Architecture')
parser.add_argument('--save_dir', help='directory for saving checkpoint', default='checkpoint.pth',type = str)
parser.add_argument('--learning_rate', help='learning rate during learning', type=float, default=0.001)
parser.add_argument('--dropout', help='dropout during learning', type=float, default=0.5)
parser.add_argument('--hidden_units', type = int, default = 500, help = 'Hidden Layer')
parser.add_argument('--epochs', help='Number of epochs for training', default=5, type=int)
parser.add_argument('--gpu', help='Enable GPU', type = str, default = 'cuda')
args = parser.parse_args()

# Describe directories relative to working directory
# Image data directories
data_dir = './flowers/'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'
        
#def main():  
# Transforms for the training, validation, and testing sets
data_transforms_training, data_transforms_validation, data_transforms_testing = pros.data_transforms()

# Load the datasets with ImageFolder
image_datasets_training, image_datasets_validation, image_datasets_testing = pros.load_datasets(train_dir, data_transforms_training, valid_dir,    data_transforms_validation, test_dir, data_transforms_testing)   

    
# TODO: Using the image datasets and the trainforms, define the dataloaders
dataloaders_training = torch.utils.data.DataLoader(image_datasets_training, batch_size=32,shuffle=True)  
# Get train data (X,y)
t_images, t_labels = next(iter(dataloaders_training))
dataloaders_validation = torch.utils.data.DataLoader(image_datasets_validation, batch_size=32)
# Get validation data (X,y)
v_images, v_labels = next(iter(dataloaders_validation))
dataloaders_testing = torch.utils.data.DataLoader(image_datasets_testing, batch_size=32)
# Get testing data (X,y)
s_images, s_labels = next(iter(dataloaders_testing))

 # Build and train the neural network (Transfer Learning)
if args.arch == 'densenet121':
    input_size = 1024
    model = models.densenet121(pretrained=True)
elif args.arch == 'vgg16':
    input_size = 25088
    model = models.vgg16(pretrained=True)    
print(model)   
# Freeze parameters so we don't backprop through them
for param in model.parameters():
  param.requires_grad = False

# Create the network, define the criterion and optimizer    
model.classifier = nn.Sequential(OrderedDict([('fc1', nn.Linear(input_size, args.hidden_units)),
                                               ('relu', nn.ReLU()),
                                               ('drop', nn.Dropout(p=args.dropout)),
                                               ('fc2', nn.Linear(args.hidden_units, 102)),
                                               ('output', nn.LogSoftmax(dim=1))]))

print(model.classifier)
#losses function:
loss_fn = nn.NLLLoss()
#learning rate
optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)      
pros.train_classifier(model, optimizer, loss_fn, args.epochs, t_images, t_labels, v_images, v_labels, args.gpu)
pros.test(model, s_images, s_labels, args.gpu)
pros.save_checkpoint(model, image_datasets_training, input_size,args.arch)
  

