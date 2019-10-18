# Imports here
import torch
from torchvision import datasets, transforms , models
from torch import nn, optim
import torch.nn.functional as F
from collections import OrderedDict
from PIL import Image
#import matplotlib as plt
#from matplotlib import pyplot as plt
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import json

def data_transforms():
    # TODO: Define your transforms for the training, validation, and testing sets
     data_transforms_training = transforms.Compose([transforms.RandomRotation(30),
                                                    transforms.RandomResizedCrop(224),
                                                    transforms.RandomHorizontalFlip(),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])
                                     

     data_transforms_validation = transforms.Compose([transforms.Resize(255),
                                                      transforms.CenterCrop(224),
                                                      transforms.ToTensor(),
                                                      transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])
                                     

     data_transforms_testing = transforms.Compose([transforms.Resize(255),
                                                   transforms.CenterCrop(224),
                                                   transforms.ToTensor(),
                                                   transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])                                    
     return data_transforms_training, data_transforms_validation, data_transforms_testing


# TODO: Load the datasets with ImageFolder
def load_datasets(train_dir, data_transforms_training, valid_dir, data_transforms_validation, test_dir, data_transforms_testing):
     image_datasets_training = datasets.ImageFolder(train_dir, transform=data_transforms_training)
     image_datasets_validation = datasets.ImageFolder(valid_dir, transform=data_transforms_validation)
     image_datasets_testing = datasets.ImageFolder(test_dir, transform=data_transforms_testing)
     return image_datasets_training, image_datasets_validation, image_datasets_testing


# Train the classifier
def train_classifier(model, optimizer, loss_fn, epochs_num, t_images, t_labels, dataloaders_training, v_images, v_labels, dataloaders_validation, device):
 #gpu:
   model.to(device);
    #Train the classifier layers using backpropagation using the pre-trained network to get the features
   epochs = epochs_num
   steps = 0
   running_loss = 0
   print_every = 32
   for epoch in range(epochs):
        # Move images and label tensors to the default gup
      for t_images, t_labels in dataloaders_training:
          steps+=1   
          t_images, t_labels = t_images.to(device), t_labels.to(device)
           #Clear the previous gradients:
          optimizer.zero_grad()
        #Feedforawrd
          pred = model.forward(t_images)
          loss = loss_fn(pred, t_labels)
    #backbropgation: train the weights
          loss.backward()
    #Adjust/update the new wights:
          optimizer.step()
          running_loss += loss.item()
        
          if steps % print_every == 0:
             validation_loss = 0
             accuracy = 0
             model.eval()# set model with out dropout:        
             with torch.no_grad(): 
        # Move images and label tensors to the default gup
               for v_images, v_labels in dataloaders_validation:
                   v_images, v_labels = v_images.to(device), v_labels.to(device)
                   pred=model(v_images)
                   v_loss=loss_fn(pred, v_labels)
                   validation_loss += v_loss.item()
                   #accurcy:
                   ps = torch.exp(model(v_images))
                   top_p, top_class= ps.topk(1)
                   equals= (top_class==v_labels.reshape(*top_class.shape))
                   accuracy = torch.mean(equals.type(torch.FloatTensor))

               print(f"Epoch {epoch+1}/{epochs}.. "
                     f"Train loss: {running_loss/print_every:.3f}.. "
                     f"validation loss: {validation_loss/len(dataloaders_validation):.3f}.. "
                     f"validation accuracy: {accuracy/len(dataloaders_validation):.3f}")
               running_loss = 0
               model.train()   
              
 # TODO: Do validation on the test set 
def test(model, s_images, s_labels, dataloaders_testing, device):
    with torch.no_grad():
       #loss
        # set model with out dropout:
        model.eval()
        # Move images and label tensors to the default gup
        for s_images, s_labels in dataloaders_testing:
            s_images, s_labels = s_images.to(device), s_labels.to(device)
        #accurcy:
            ps = torch.exp(model(s_images))
            top_p, top_class= ps.topk(1)
            equals= (top_class==s_labels.reshape(*top_class.shape))
            accuracy = torch.mean(equals.type(torch.FloatTensor))
    print(f'Accuracy of the network on the test set: {accuracy.item()*100}%')
        
#saving the model checkpoint
def save_checkpoint(model, image_datasets_training, input_size,arch,save_dir):
    model.class_to_idx = image_datasets_training.class_to_idx
    checkpoint = {'input_size': input_size,
              'output_size': 102,
              'state_dict': model.state_dict(),
              'class_to_idx':model.class_to_idx,
              'classifier': model.classifier,
               'model_name': arch
             }

    torch.save(checkpoint, save_dir) 
    print("done save checkpoint")
#load checkpoint
def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
        # Download pretrained model i used before 
    if checkpoint['model_name'] == 'densenet121':
        model = models.densenet121(pretrained=True)        
    elif checkpoint['model_name'] == 'vgg16':  
        model = models.vgg16(pretrained=True)
    # Freeze parameters
    for parameter in model.parameters():
        parameter.requires_grad = False

    # Load parameters from checkpoint
    model.class_to_idx = checkpoint['class_to_idx']
    model.classifier =checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict']) 
    model.input_size  = checkpoint['input_size']
    model.output_size =checkpoint['output_size']
  
    return model
#Image Preprocessing
def process_image(image):
    img_test = Image.open(image)
   
    adjustments = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
                                     ])     
    img_tensor = adjustments(img_test)
    
    return img_tensor

def imshow(image, ax=None, title=None):
  if ax is None:
    fig, ax = plt.subplots()    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.numpy().transpose((1, 2, 0))    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean   
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)    
    ax.imshow(image)
    
    return ax

# Load class_to_name json file 
def load_json(json_file):
    with open(json_file, 'r') as f:
        flower_to_name = json.load(f)
        return flower_to_name
    
def predict(image_path, model, topk, device):  
    model.to(device)
    img_torch = process_image(image_path)
    img_torch = img_torch.unsqueeze_(0)
    
    with torch.no_grad():
        model.to(img_torch)
        output = model(img_torch)
    # Calculating probabilities
    probs = torch.exp(output)
    probs_top = probs.topk(topk)[0]
    index_top = probs.topk(topk)[1]
    
    # Converting probabilities and outputs to lists
    probs_top_list = np.array(probs_top)[0]
    index_top_list = np.array(index_top[0])
    
    # Loading index and class mapping
    class_to_idx = model.class_to_idx
    # Inverting index-class dictionary
    indx_to_class = {x: y for y, x in class_to_idx.items()}

    # Converting index list to class list
    classes_top_list = []
    for index in index_top_list:
        classes_top_list += [indx_to_class[index]]
    return  list(probs_top_list), classes_top_list

# TODO: Display an image along with the top 5 classes
# Define image path
def display_image(image_path, class_to_name, classes,probs):
   imshow(process_image(image_path))
# Converting classes to names
   names = []
   for i in classes:
     names += [class_to_name[i]]

   plt.figure(figsize = (6,10)) 
   plt.subplot(2,1,2)
   sns.barplot(x=probs, y=names);  
   plt.show()
   return print(names)