# Image Classifier Project

### Project Description

This project is a part of Udacity Data Scientist Nanodegree Program. 
In this project, I have train an image classifier to recognize different species of flowers. 
You can imagine using something like this in a phone app that tells you the name of the flower your camera is looking at. 
The project is broken down into multiple steps:
- Load and preprocess the image dataset
- Train the image classifier on your dataset
- Use the trained classifier to predict image content

### Data:

The data used are a flower database, It is not provided in the repository as it's larger than what GitHub allows, but you can download it from Image Classifier Project.ipynb.   
The data is divided into 3 folders:

- test
- train
- validate

### Software and Libraries:

- Python
- torch
- torchvision
- collections
- PIL 
- matplotlib
- numpy
- pandas
- seaborn
- As this project uses deep learning, for training of network you need to use a GPU.

### Instructions:

The project has divided into two part:
1. develop the model through a Jupyter notebook.
- Classifier Project.ipynb
2. convert it into an application by used Python scripts that run from the command line.
- there are 3 files:
   1. train : train a new network on a dataset and save the model as a checkpoint.
   2. predict : uses a trained network to predict the class for an input image.
   3. pros : contains the functions.


### Command Line Application:
1. Train a new network on a data set with train.py

- Basic usage: python train.py data_directory
- Prints out training loss, validation loss, and validation accuracy as the network trains
##### Options:
- Set directory to save checkpoints: python train.py data_dir --save_dir save_directory
- Choose architecture: python train.py data_dir --arch "vgg13"
- Set hyperparameters: python train.py data_dir --learning_rate 0.01 --hidden_units 512 --epochs 20
- Use GPU for training: python train.py data_dir --gpu

2. Predict flower name from an image with predict.py along with the probability of that name. That is, you'll pass in a single image /path/to/image and return the flower name and class probability.

- Basic usage: python predict.py /path/to/image checkpoint
##### Options:
- Return top KK most likely classes: python predict.py input checkpoint --top_k 3
- Use a mapping of categories to real names: python predict.py input checkpoint --category_names cat_to_name.json
- Use GPU for inference: python predict.py input checkpoint --gpu
