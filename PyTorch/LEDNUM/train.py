from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import TensorDataset, DataLoader
from torchvision import transforms
from torch import optim
from model import LeNet
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.nn as nn
import torch
import time


#通过Pytorch的Dataset类,自定义加载自己的数据集-----可以作为通过
class MyDataset(Dataset):

    def __init__(self, txt_path, transform=None, target_transform=None):
        fh = open(txt_path, 'r')
        imgs = []
        for line in fh:
            line = line.rstrip()
            words = line.split()
            imgs.append((words[0], int(words[1])))
            self.imgs = imgs
            self.transform = transform
            self.target_transform = target_transform

    def __getitem__(self, index):
        fn, label = self.imgs[index]
        #img = Image.open(fn).convert('RGB')
        img = Image.open(fn)
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.imgs)

    #创建模型，部署gpu


def train_runner(model, device, trainloader, optimizer, epoch):
    #训练模型, 启用 BatchNormalization 和 Dropout, 将BatchNormalization和Dropout置为True
    model.train()
    total = 0
    correct = 0.0

    #enumerate迭代已加载的数据集,同时获取数据和数据下标
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        #把模型部署到device上
        inputs, labels = inputs.to(device), labels.to(device)
        #初始化梯度
        optimizer.zero_grad()
        #保存训练结果
        outputs = model(inputs)
        #计算损失和
        #多分类情况通常使用cross_entropy(交叉熵损失函数), 而对于二分类问题, 通常使用sigmod
        loss = F.cross_entropy(outputs, labels)
        #获取最大概率的预测结果
        #dim=1表示返回每一行的最大值对应的列下标
        predict = outputs.argmax(dim=1)
        total += labels.size(0)
        correct += (predict == labels).sum().item()
        #反向传播
        loss.backward()
        #更新参数
        Loss = []
        Accuracy = []
        optimizer.step()
        if i % 100 == 0:
            #loss.item()表示当前loss的数值
            print("Train Epoch{} \t Loss: {:.6f}, accuracy: {:.6f}%".format(epoch, loss.item(),
                                                                            100 * (correct / total)))
            Loss.append(loss.item())
            Accuracy.append(correct / total)
    return loss.item(), correct / total


def test_runner(model, device, testloader):
    #模型验证, 必须要写, 否则只要有输入数据, 即使不训练, 它也会改变权值
    #因为调用eval()将不启用 BatchNormalization 和 Dropout, BatchNormalization和Dropout置为False
    model.eval()
    #统计模型正确率, 设置初始值
    correct = 0.0
    test_loss = 0.0
    total = 0
    #torch.no_grad将不会计算梯度, 也不会进行反向传播
    with torch.no_grad():
        for data, label in testloader:
            data, label = data.to(device), label.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, label).item()
            predict = output.argmax(dim=1)
            #计算正确数量
            total += label.size(0)
            correct += (predict == label).sum().item()
        #计算损失值
        print("test_avarage_loss: {:.6f}, accuracy: {:.6f}%".format(test_loss / total, 100 * (correct / total)))


if __name__ == '__main__':
    # 预处理
    pipline_train = transforms.Compose([
        #随机旋转图片
        transforms.RandomHorizontalFlip(),
        #将图片尺寸resize到32x32
        transforms.Resize((32, 32)),
        #将图片转化为Tensor格式
        transforms.ToTensor(),
        #正则化(当模型出现过拟合的情况时，用来降低模型的复杂度)
        transforms.Normalize((0.1307, ), (0.3081, ))
    ])
    pipline_test = transforms.Compose([
        #将图片尺寸resize到32x32
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307, ), (0.3081, ))
    ])
    train_data = MyDataset('./DataSet/LEDNUM/train.txt', transform=pipline_train)
    test_data = MyDataset('./DataSet/LEDNUM/test.txt', transform=pipline_test)

    #train_data 和test_data包含多有的训练与测试数据，调用DataLoader批量加载
    trainloader = torch.utils.data.DataLoader(dataset=train_data, batch_size=32, shuffle=True)
    testloader = torch.utils.data.DataLoader(dataset=test_data, batch_size=8, shuffle=False)

    # 部署到GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LeNet().to(device)
    #定义优化器
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    #调用
    epoch = 200
    Loss = []
    Accuracy = []
    for epoch in range(1, epoch + 1):
        print("start_time", time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
        loss, acc = train_runner(model, device, trainloader, optimizer, epoch)
        Loss.append(loss)
        Accuracy.append(acc)
        test_runner(model, device, testloader)
        print("end_time: ", time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())), '\n')

    print('Finished Training')
    plt.subplot(2, 1, 1)
    plt.plot(Loss)
    plt.title('Loss')
    plt.show()
    plt.subplot(2, 1, 2)
    plt.plot(Accuracy)
    plt.title('Accuracy')
    plt.show()

    torch.save(model, './PyTorch/LEDNUM/model-mine.pth')  #保存模型