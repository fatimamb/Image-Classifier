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
import matplotlib.pyplot
import pros 
import json
import argparse
parser = argparse.ArgumentParser(description='Image Classifier Predictions')

# Command line arguments
parser.add_argument('--image_dir', type = str, default = 'flowers/test/25/image_06583.jpg', help = 'Path to image')
parser.add_argument('--checkpoint', type = str, default = 'checkpoint.pth', help = 'Path to checkpoint')
parser.add_argument('--topk', type = int, default = 5, help = 'Top k classes and probabilities')
parser.add_argument('--jsonfile', type = str, default = 'cat_to_name.json', help = 'class_to_name json file')
parser.add_argument('--gpu', type = str, default = 'cuda', help = 'GPU or CPU')

args = parser.parse_args()

# Load in a mapping from category label to category name
class_to_name = pros.load_json(args.jsonfile)

# Load pretrained network
model = pros.load_checkpoint(args.checkpoint)
print(model)  

# Scales, crops, and normalizes a PIL image for the PyTorch model; returns a Numpy array
image = pros.process_image(args.image_dir)

# Display image
pros.imshow(image)

# Highest k probabilities and the indices of those probabilities corresponding to the classes (converted to the actual class labels)
probabilities, classes = pros.predict(args.image_dir, model, args.topk, args.gpu)  
print(probabilities)
print(classes)

# Display the image along with the top 5 classes
pros.display_image(args.image_dir, class_to_name, classes,probabilities)