from this import d
import torch
import os
from torch import nn
from LeNet import LeNet_5
from torch.optim import lr_scheduler
from torchvision import datasets,transforms

# 数据格式转换为tensor格式
data_transform = transforms.Compose([transforms.ToTensor()])
# 加载数据集
train_dataset = datasets.Mnist(root = "./MNIST数据集",train = True,transform = data_transform, download = False)
train_dataLoader = torch.utils.data.Downloader()
 
