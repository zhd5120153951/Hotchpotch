from model import Model
from model import LeNet
import numpy as np
import torch
import os
import time
from torchvision import datasets
from torchvision import transforms
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from matplotlib import pyplot as plt
import torch.nn.functional as F

batch_size = 1024
"""
加载数据集
"""
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307, ), (0.3081, ))
])
#方法二--加载本地数据集

#train = true是训练集，false为测试集--这种方法要每次下载数据集，可以实现本地加载
MNIST_dataset_train = datasets.MNIST(root='./LeNet/data/train', train=True, download=True, transform=transform)
dataloaders_train = DataLoader(dataset=MNIST_dataset_train, batch_size=batch_size, shuffle=True)

MNIST_dataset_test = datasets.MNIST(root="./LeNet/data/test", train=False, download=True, transform=transform)
dataloaders_test = DataLoader(dataset=MNIST_dataset_test, batch_size=batch_size, shuffle=True)

#定义优化器和设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LeNet().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

Loss = []
Accuracy = []
epochs = 100


def train_mnist(model, device, dataloaders_train, optimizer, epoch):
    model.train()
    total = 0
    correct = 0.0

    for i, data in enumerate(dataloaders_train, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = F.cross_entropy(outputs, labels)
        predict = outputs.argmax(dim=1)
        total += labels.size(0)
        correct += (predict == labels).sum().item()
        loss.backward()
        optimizer.step()

        if i % 100 == 0:
            print("train epoch {}\tloss:{:.6f},accuracy:{:.6f}%".format(epoch, loss.item(), 100 * (correct / total)))
            Loss.append(loss.item())
            Accuracy.append(correct / total)

    return loss.item(), correct / total


def test_minst(model, device, dataloaders_test):
    model.eval()
    correct = 0.0
    test_loss = 0.0
    total = 0
    with torch.no_grad():
        for data, label in dataloaders_test:
            data, label = data.to(device), label.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, label).item()
            predict = output.argmax(dim=1)
            total += label.size(0)
            correct += (predict == label).sum().item()
        print("test_average_loss:{:.6f},accuracy:{:.6f}%".format(test_loss / total, 100 * correct / total))
    return 100 * correct / total


def drawImage():
    print("finish train")
    plt.subplot(2, 1, 1)
    plt.plot(Loss)
    plt.title("Loss line")
    plt.show()

    plt.subplot(2, 1, 2)
    plt.plot(Accuracy)
    plt.title("Accuracy line")
    plt.show()


if __name__ == '__main__':
    test_acuur_temp = 0.0
    for epoch in range(1, epochs + 1):
        print("start time", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time())))
        train_loss, train_acurr = train_mnist(model, device, dataloaders_train, optimizer, epoch)
        Loss.append(train_loss)
        Accuracy.append(train_acurr)
        test_acuur = test_minst(model, device, dataloaders_test)
        if test_acuur >= test_acuur_temp:
            test_acuur_temp = test_acuur
            torch.save(model, "./LeNet/models/model-minist.pth")

        print("end time", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time())), "\n")
    drawImage()
    print(model)
