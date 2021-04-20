import cv2
import os
import time
import utils
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
import argparse

import torch
import torch.optim as optim
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset


# contruct the argument parser
# ap = argparse.ArgumentParser()
# ap.add_argument('-m', '--model', required=True,
#                 help='path to save the trained model')
# ap.add_argument('-e', '--epochs', type=int, default=100,
#                 help='number of epochs to train our network for')
# args = vars(ap.parse_args())

# learning params
lr = 1e-3
batch_size = 32

# check cuda
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Computation device: {device}')

# get clip and label
df = pd.read_csv('data.csv')
clips = df['clip'].tolist()

# print(X[0].replace("'", "").replace("[", "").replace("]", "").split(", "))
# print(y[:2])

def alter_type(clip):
    new_clip_array = clip.replace("'", "").replace("[", "").replace("]", "").split(", ")
    return new_clip_array

def get_clip_input(all_clips):
    X = []
    for each_clip in all_clips:
        each_clip = alter_type(each_clip)
        X.append(each_clip)
    return X

# get X, y
X = get_clip_input(clips)
y = df['label'].tolist()
