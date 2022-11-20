from model import Model
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

"""
加载数据集
"""  
    
if __name__ == '__main__':
    batch_size = 1024

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081, ))
    ])
    #方法二--加载本地数据集
    
    #train = true是训练集，false为测试集--这种方法要每次下载数据集，可以实现本地加载
    MNIST_dataset_train = datasets.MNIST(root='./LeNet/data/train', train=True, download=True, transform=transform)
    dataloaders_train = DataLoader(dataset=MNIST_dataset_train, batch_size=batch_size, shuffle=True)

    MNIST_dataset_test = datasets.MNIST(root="./LeNet/data/test",train=False,download=True,transform=transform)
    dataloaders_test = DataLoader(dataset= MNIST_dataset_test,batch_size=batch_size,shuffle=True)
    
    #train_dataset = datasets.MNIST(root='./train/MNIST', train=True, download= False, transform=ToTensor())
    #test_dataset = datasets.MNIST(root='./test/MNIST', train=False, download = False, transform=ToTensor())
    #train_loader = DataLoader(train_dataset, batch_size=batch_size)
    #test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    #方法二--加载本地数据集

    #加载到GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Model().to(device)
    #优化器-sgd or adam
    sgd = SGD(model.parameters(), lr=1e-1)
    optimizer = torch.optim.Adam(model.parameters(),lr=0.001)
    
    loss_fn = CrossEntropyLoss()
    all_epoch = 100
    
    for current_epoch in range(all_epoch):
        model.train()
        loss_temp = 0
        for idx, (train_x, train_label) in enumerate(dataloaders_train):
            # sgd.zero_grad()
            optimizer.zero_grad()
            predict_y = model(train_x.to(device))
            loss = loss_fn(predict_y, train_label.to(device))
            if loss > loss_temp:
                loss_temp = loss
            else:
                loss_temp = loss
            if idx % 10 == 0:
                print('idx: {}, loss: {}'.format(idx, loss_temp.sum().item()))
            loss.backward()
            #sgd.step()
            optimizer.step()
            
        '''
        all_correct_num = 0
        all_sample_num = 0
        model.eval()
        for idx, (test_x, test_label) in enumerate(dataloaders_test):
            predict_y = model(test_x).detach()
            predict_y = np.argmax(predict_y, axis=-1)
            current_correct_num = predict_y == test_label
            all_correct_num += np.sum(current_correct_num.numpy(), axis=-1)
            all_sample_num += current_correct_num.shape[0]
        acc = all_correct_num / all_sample_num
        print('accuracy: {:.4f}'.format(acc))
        torch.save(model, 'models/mnist_{:.4f}.pth'.format(acc))
        '''