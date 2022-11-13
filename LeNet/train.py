from model import Model
import numpy as np
import torch
from torchvision.datasets import mnist
from torchvision import transforms
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor

if __name__ == '__main__':
    batch_size = 256

    train_dataset = mnist.MNIST(root='./train', train=True, transform=ToTensor())
    test_dataset = mnist.MNIST(root='./test', train=False, transform=ToTensor())
    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
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
        for idx, (train_x, train_label) in enumerate(train_loader):
            sgd.zero_grad()
            predict_y = model(train_x.float())
            loss = loss_fn(predict_y, train_label.long())
            if idx % 10 == 0:
                print('idx: {}, loss: {}'.format(idx, loss.sum().item()))
            loss.backward()
            sgd.step()

        all_correct_num = 0
        all_sample_num = 0
        model.eval()
        for idx, (test_x, test_label) in enumerate(test_loader):
            predict_y = model(test_x.float()).detach()
            predict_y = np.argmax(predict_y, axis=-1)
            current_correct_num = predict_y == test_label
            all_correct_num += np.sum(current_correct_num.numpy(), axis=-1)
            all_sample_num += current_correct_num.shape[0]
        acc = all_correct_num / all_sample_num
        print('accuracy: {:.2f}'.format(acc))
        torch.save(model, 'models/mnist_{:.2f}.pkl'.format(acc))
