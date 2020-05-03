#!/usr/bin/env python3
#
# PROGRAMMER: Mufeed H. AlMatar
# DATE CREATED: 02 May 2020                                  
# REVISED DATE: 
# PURPOSE: Create a function that parses command line inputs for training and predicting
#          from the user using the Argparse Python module. If the user fails to 
#          provide some or all of the inputs, then the default values are
#          used for the missing inputs. 
# 
#     train.py Command Line Arguments:
#     1. Set directory to save checkpoints: 
#         python train.py data_dir --save_dir save_directory
#     2. Choose architecture: 
#         python train.py data_dir --arch "vgg"
#     3. Set hyperparameters: 
#         python train.py data_dir --learning_rate 0.01 --hidden_units 512 --epochs 20
#     4. Use GPU for training: 
#         python train.py data_dir --gpu
# 
#     predict.py Command Line Arguments:
#     1. Return top K most likely classes: 
#           python predict.py input checkpoint --top_k 3
#     2. Use a mapping of categories to real names: 
#           python predict.py input checkpoint --category_names cat_to_name.json
#     3. Use GPU for inference: 
#           python predict.py input checkpoint --gpu
#
import argparse
from model_helper import Phase

def parse_args(phase = Phase.train):
  if phase == Phase.train:
    parser = argparse.ArgumentParser(
      description='Train a new network on a dataset and save the model as a checkpoint',
    )

    parser.add_argument('data_directory', type=str, default='flowers', \
      help='Trainging data directory')
    parser.add_argument('--save_dir', type=str, default='checkpoit.pth', \
      help='Set directory to save checkpoints')
    parser.add_argument('--arch', choices=['vgg', 'densenet', 'alexnet'], default='densenet', \
      help='Choose architecture')
    parser.add_argument('--learning_rate', type=float, default=0.01, \
      help='Learning rate')
    parser.add_argument('--hidden_units', type=int, default=512, \
      help='Hidden Layer Units')
    parser.add_argument('--epochs', type=int, default=5, \
      help='Training epochs')
    parser.add_argument('--accurecy', type=int, default=100, \
      help='Stop the training if the network reached provided accurecy')
    parser.add_argument('--gpu', action="store_true", default=False, \
      help='Use GPU for training')
    
    return parser.parse_args()
  else:
    parser = argparse.ArgumentParser(
      description='Predict flower name from an image along with the probability of that name.',
    )

    parser.add_argument('image_path', type=str, help='Path for predicted image')
    parser.add_argument('--top_k', type=int, default=5, \
      help='Return top K most likely classes')
    parser.add_argument('--category_names', type=str, default='cat_to_name.json', \
      help='Use a mapping of categories to real names')
    parser.add_argument('--gpu', action="store_true", default=False, \
      help='Use GPU for inference')
    parser.add_argument('checkpoint', type=str, help='Checkpoint file path')
    
    return parser.parse_args()    
