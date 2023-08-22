import numpy as np
import torch
import os
from torchvision.datasets import mnist
from torchvision import datasets
from torchvision import transforms
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from torch import nn
import torch.nn.functional as F
from torch import optim
import matplotlib.pyplot as plt
import time
from module import LeNetRGB,train_runner,test_runner



pipline_train = transforms.Compose([
    #随机旋转图片
    transforms.RandomHorizontalFlip(),
    #将图片尺寸resize到32x32
    transforms.Resize((32,32)),
    #将图片转化为Tensor格式
    transforms.ToTensor(),
    #正则化(当模型出现过拟合的情况时，用来降低模型的复杂度)
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
pipline_test = transforms.Compose([
    #将图片尺寸resize到32x32
    transforms.Resize((32,32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
#下载数据集
train_set = datasets.CIFAR10(root="./data/train", train=True, download=True, transform=pipline_train)
test_set = datasets.CIFAR10(root="./data/test", train=False, download=True, transform=pipline_test)
#加载数据集
# dataloaders_train = DataLoader(dataset=MNIST_dataset_train, batch_size=batch_size, shuffle=True)
trainloader = DataLoader(dataset = train_set, batch_size  = 1024, shuffle = True)
testloader = DataLoader(dataset = test_set, batch_size = 512, shuffle = True)
# 类别信息也是需要我们给定的
classes = ('plane', 'car', 'bird', 'cat','deer', 'dog', 'frog', 'horse', 'ship', 'truck')


#创建模型，部署gpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LeNetRGB().to(device)
#定义优化器
optimizer = optim.Adam(model.parameters(), lr=0.001)
        
if __name__ == '__main__':
    #调用
    epoch = 100
    Loss = []
    Accuracy = []
    ls = []
    accur = []
    for epoch in range(1, epoch+1):
        print("start_time",time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
        loss, acc = train_runner(model, device, trainloader, optimizer, epoch, Loss, Accuracy)
        #Loss.append(loss)
        #Accuracy.append(acc)
        ls.append(loss)
        accur.append(acc)
        test_runner(model, device, testloader)
        print("end_time: ",time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())),'\n')
    
    
    print('Finished Training')
    plt.subplot(2,1,1)
    plt.plot(ls)
    plt.title('Loss')
    plt.show()
    plt.subplot(2,1,2)
    plt.plot(accur)
    plt.title('Accuracy')
    plt.show()

    print(model)
    torch.save(model, './models/model-cifar10.pth') #保存模型