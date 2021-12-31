import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import cv2
import torch
import torchvision
import os
import shutil
from PIL import Image
from torch.nn import functional as F
import  torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets.utils import download_url
import math
import datetime
import argparse
from sklearn.model_selection import train_test_split
from torch import optim
from torchvision import models

device='cuda:0'

class PetData(torch.utils.data.Dataset):
    def __init__(self, data, train_csv):
        self.data = data
        self.index = list(data.keys())
        self.train_csv = train_csv

    def __len__(self):
        return self.data.__len__()

    def __getitem__(self, idx):
        Id = self.index[idx]
        correct = self.train_csv[self.train_csv['Id'] == Id]['Pawpularity'].to_numpy()
        return self.data[Id], correct

class NeuralNet(nn.Module):
    def __init__(self, in_size):
        super(NeuralNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size = (3, 3), stride = (2, 2))
        self.conv2 = nn.Conv2d(16, 32, kernel_size = (3, 3), stride = (2, 2))
        self.conv3 = nn.Conv2d(32, 128, kernel_size = (3, 3), stride = (2, 2))
        after_conv1_size = self.size(in_size[0], in_size[1], (1, 1), (0, 0), (3, 3), (2, 2))
        after_conv2_size = self.size(after_conv1_size[0], after_conv1_size[1], (1, 1), (0, 0), (3, 3), (2, 2))
        after_conv3_size = self.size(after_conv2_size[0], after_conv2_size[1], (1, 1), (0, 0), (3, 3), (2, 2))
        self.fc1 = nn.Linear(after_conv3_size[0] * after_conv3_size[1] * 128, 2048)
        self.fc2 = nn.Linear(2048, 1)
        self.fc3 = nn.Linear(128, 1)
        self.dropout1 = torch.nn.Dropout2d(p=0.3)
        self.dropout2 = torch.nn.Dropout(p=0.3)
        self.resnet50 = models.resnet50(pretrained=True)
        self.resnet50.fc = self.fc2

    def size(self, hin, win, dilation, padding, kernel_size, stride):

        hout = math.floor((hin + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) / stride[0] + 1)
        wout = math.floor((win + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) / stride[1] + 1)
        return hout, wout
    
    def forward(self, x):
        # out = torch.relu(self.conv1(x))
        # out = self.dropout1(out)
        # out = torch.relu(self.conv2(out))
        # print(out.shape)
        # out = self.dropout1(out)
        # out = torch.relu(self.conv3(out))
        # out = self.dropout1(out)
        # out = out.reshape(out.shape[0], -1)
        # # out = out.view(out.shape[0], -1)/
        out = self.resnet50(x)
        # out = torch.relu(self.fc1(out))
        # out = self.dropout2(out)
        # out = torch.relu(self.fc2(out))
        
        return out

def ArgParser():
    parser = argparse.ArgumentParser(description='このプログラムの説明（なくてもよい）')    # 2. パーサを作る

    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--store_name')

    args = parser.parse_args() 

    return args

def train(n_epochs, optimizer, model, loss_fn, alpha, train_data, valid_data, batch_size, data_width, file_path):

    train_loss_list = []
    val_loss_list = []

    for epoch in range(1, n_epochs + 1):
        train_loss_sum = 0.0
        train_accuracy_sum = 0
        train_cnt = 0
        
        model.train()
        
        for imgs, labels in train_data:
            
            imgs = torch.reshape(imgs, (-1, 3, data_width[0], data_width[1])).to(device=device, dtype=torch.float)
            labels = labels.to(device=device, dtype=torch.float)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = loss_fn(outputs, labels)
            l2 = torch.tensor(0., requires_grad=True)
            for w in model.parameters():
                l2 = l2 + torch.norm(w)**2
            loss = loss + alpha*l2
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item()
            train_cnt += labels.shape[0]
            imgs.to(device='cpu')


        model.eval()
        val_loss_sum = 0.0
        val_accuracy_sum = 0
        val_cnt = 0
        
        with torch.no_grad():
            for imgs, labels in valid_data:
                imgs = torch.reshape(imgs, (-1, 3, data_width[0], data_width[1])).to(device=device, dtype=torch.float)
                labels = labels.to(device=device)
                outputs = model(imgs)
                loss = loss_fn(outputs, labels)
                val_loss_sum += loss.item()
                val_cnt += labels.shape[0]
                imgs.to(device='cpu')
        train_loss = train_loss_sum / train_cnt
        val_loss = val_loss_sum / val_cnt
        
        train_loss_list.append(train_loss)
        val_loss_list.append(val_loss)

        print('{} Epoch : {:>4}, Train Loss : {:.8f}, Val Loss : {:.8f}'.format(datetime.datetime.now(), epoch, train_loss, val_loss))

    x = np.arange(1, n_epochs + 1)
    plt.plot(x, train_loss_list)
    plt.plot(x, val_loss_list)
    plt.xlabel("epoch")
    plt.ylabel("mse loss")
    plt.savefig(file_path + '/loss.png')
    return model

def Resize(img, size):
    height, width, color = img.shape # 画像の縦横サイズを取得

    if height > size[0] or width > size[1]:
        raise print("Size Error")
    
    padding_top = 0
    padding_left = 0
    padding_right = 0
    padding_bottom = 0
        
      # 縦長画像→幅を拡張する
    if height < size[0]:
        height_diffsize = size[0] - height
        # 元画像を中央ぞろえにしたいので、左右に均等に余白を入れる
        padding_top = int(height_diffsize / 2)
        padding_bottom = height_diffsize - padding_top

    if width < size[1]:
        width_diffsize = size[1] - width
        padding_left = int(width_diffsize / 2)
        padding_right = width_diffsize - padding_left
    padding_img = cv2.copyMakeBorder(img, padding_top, padding_bottom, padding_left, padding_right, cv2.BORDER_REPLICATE)

        
    return padding_img



def main():
    args = ArgParser()

    dirname=input("Input Directory Name For Result : ")
    model_path = './model/' + dirname
    os.makedirs(model_path)
    shutil.copyfile('./train.py',model_path + '/train_copy.py')
    discription = input("Description : ")


    print("Load Data")
    train_csv = pd.read_csv('./data/train.csv')
    test_csv = pd.read_csv('./data/test.csv')
    gender_submission = pd.read_csv('./data/sample_submission.csv')
    # test_data = pd.merge(test, gender_submission, how="inner", on="PassengerId")

    img_list = {}
    mode = 'train'
    size = [1280, 1280]
    input_size = [224, 224]
    print("Load Image Data")
    for img_name in train_csv['Id']:
        img = cv2.imread('./data/' + mode + '/' + img_name +'.jpg')
        img_list[img_name] = cv2.resize(Resize(img, size) , (int(input_size[0]), int(input_size[1])))
    
    print("Formatting Data")
    pet_dataloader = PetData(img_list, train_csv)
    train_data, valid_data = train_test_split(pet_dataloader, test_size=0.2)
    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)
    valid_dataloader = torch.utils.data.DataLoader(valid_data, batch_size=32, shuffle=True) 


    model = NeuralNet(in_size=input_size).to(device=device)
    lr = 1e-2
    alpha = 0.01
    optimizer = optim.Adam(model.parameters(), lr)
    loss_fn = nn.MSELoss()
    print(model)
    print("Learning Rate : {}".format(lr))
    print("Start Train")
    trained_model = train(200, optimizer, model, loss_fn, alpha, train_dataloader, valid_dataloader, 16, input_size, model_path)
    
    torch.save(model.state_dict(), model_path + '/model.pth')

if __name__ == '__main__':
    main()
