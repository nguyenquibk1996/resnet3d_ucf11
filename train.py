import cv2
import os
import time
import utils
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
import argparse
from PIL import Image

import torchvision
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset


# contruct the argument parser
ap = argparse.ArgumentParser()
ap.add_argument('-m', '--model', required=True,
                help='path to save the trained model')
ap.add_argument('-e', '--epochs', type=int, default=100,
                help='number of epochs to train our network for')
args = vars(ap.parse_args())


# check cuda
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Computation device: {device}')

# get clip and label
df = pd.read_csv('data.csv')
clips = df['clip'].tolist()

# change string type to array type
def alter_type(clip):
    new_clip_array = clip.replace("'", "").replace("[", "").replace("]", "").replace("frames_temp", "frames_UCF11").split(", ")
    return new_clip_array

# process all clip_len frames to get X 
def get_clip_input(all_clips):
    X = []
    for each_clip in all_clips:
        each_clip = alter_type(each_clip)
        X.append(each_clip)
    return X

# get X, y
X = get_clip_input(clips)
y = df['label'].tolist()
print('length X: ', len(X))
print('length y: ', len(y))

# train, test split
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.2, random_state=42)
print(f'Training instances: {len(X_train)}')
print(f'Validataion instances: {len(X_test)}')

# custom dataset
class UCF11(Dataset):
    def __init__(self, clips, labels):
        self.clips = clips
        self.labels = labels

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, i):
        clip = self.clips[i]
        input_frames = []
        for frame in clip:
            image = Image.open(frame)
            image = image.convert('RGB')
            image = np.array(image)
            image = utils.transforms(image=image)['image']
            input_frames.append(image)
        input_frames = np.array(input_frames)
        # print('input_frames.shape: ', input_frames.shape)
        # input_frames = np.expand_dims(input_frames, axis=0)
        input_frames = np.transpose(input_frames, (3,0,1,2))
        input_frames = torch.tensor(input_frames, dtype=torch.float32)
        input_frames = input_frames.to(device)
        # label
        self.labels = np.array(self.labels)
        lb = LabelBinarizer()
        self.labels = lb.fit_transform(self.labels)
        label = self.labels[i]
        label = torch.tensor(label, dtype=torch.long)
        # print('label: ', label)
        label = label.to(device)
        return (input_frames, label)

train_data = UCF11(X_train, y_train)
val_data = UCF11(X_test, y_test)

# learning params
lr = 1e-3
batch_size = 32

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True)

# model
model = torchvision.models.video.r3d_18(pretrained=True, progress=True)
for param in model.parameters():
    param.requires_grad = False
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 11)
model = model.to(device)
print(model)

# criterion
criterion = nn.CrossEntropyLoss()

# optim
optimizer = optim.SGD(model.fc.parameters(), lr=lr, momentum=0.9)

# scheduler
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.5,
    patience=5,
    min_lr=1e-6,
    verbose=True
)

# training
def fit(model, train_dataloader):
    print('Training')
    model.train()
    train_running_loss = 0.0
    train_running_correct = 0
    for i, data in tqdm(enumerate(train_dataloader), total=int(len(train_data)/train_dataloader.batch_size)):
        data, target = data[0].to(device), data[1].to(device)
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, torch.max(target, 1)[1])
        train_running_loss += loss.item()
        _, preds = torch.max(outputs.data, 1)
        train_running_correct += (preds == torch.max(target, 1)[1]).sum().item()
        loss.backward()
        optimizer.step()
    
    train_loss = train_running_loss/len(train_dataloader.dataset)
    train_accuracy = 100. * train_running_correct/len(train_dataloader.dataset)

    print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}')

    return train_loss, train_accuracy

# validation
def validate(model, test_dataloader):
    print('Validating')
    model.eval()
    val_running_loss = 0.0
    val_running_correct = 0
    with torch.no_grad():
        for i, data in tqdm(enumerate(test_dataloader), total=int(len(test_data)/test_dataloader.batch_size)):
            data, target = data[0].to(device), data[1].to(device)
            outputs = model(data)
            loss = criterion(outputs, torch.max(target, 1)[1])

            val_running_loss += loss.item()
            _, preds = torch.max(outputs.data, 1)
            val_running_correct += (preds == torch.max(target, 1)[1]).sum().item()

        val_loss = val_running_loss/len(test_dataloader.dataset)
        val_accuracy = 100. * val_running_correct/len(test_dataloader.dataset)
        print(f'Val loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}')

        return val_loss, val_accuracy


train_loss, train_accuracy = [], []
val_loss, val_accuracy = [], []
start = time.time()
for epoch in range(args['epochs']):
    print(f"Epoch {epoch+1} of {args['epochs']}")
    train_epoch_loss, train_epoch_accuracy = fit(model, train_loader)
    val_epoch_loss, val_epoch_accuracy = validate(model, test_loader)
    train_loss.append(train_epoch_loss)
    train_accuracy.append(train_epoch_accuracy)
    val_loss.append(val_epoch_loss)
    val_accuracy.append(val_epoch_accuracy)
    scheduler.step(val_epoch_loss)

end = time.time()

print(f'{(end-start)/60:.3f} minutes')

# accuracy plots
plt.figure(figsize=(10, 7))
plt.plot(train_accuracy, color='green', label='train accuracy')
plt.plot(val_accuracy, color='blue', label='validataion accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('outputs/accuracy.png')
# loss plots
plt.figure(figsize=(10, 7))
plt.plot(train_loss, color='orange', label='train loss')
plt.plot(val_loss, color='red', label='validataion loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('outputs/loss.png')
    
# serialize the model to disk
print('Saving model...')
torch.save(model.state_dict(), args['model'])
 
print('TRAINING COMPLETE')