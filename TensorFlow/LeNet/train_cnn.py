'''
@FileName   :train_cnn.py
@Description:cnn模型训练代码，训练的代码会保存在models目录下，折线图会保存在results目录下
@Date       :2023/03/04 10:25:26
@Author     :daito
@Website    :Https://github.com/zhd5120153951
@Copyright  :daito
@License    :None
@version    :1.0
@Email      :2462491568@qq.com
'''

import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import preprocessing as ppcs
from tensorflow.keras import Sequential, layers, optimizers, datasets
from tensorflow import data
from time import *


# 数据集加载函数，指明数据集的位置并统一处理为imgheight*imgwidth的大小，同时设置batch
def data_load(data_dir, test_data_dir, img_height, img_width, batch_size):
    # 加载训练集
    train_data = ppcs.image_dataset_from_directory(data_dir,
                                                   label_mode='categorical',
                                                   seed=123,
                                                   image_size=(img_height, img_width),
                                                   batch_size=batch_size)
    # 加载测试集
    val_data = ppcs.image_dataset_from_directory(test_data_dir,
                                                 label_mode='categorical',
                                                 seed=123,
                                                 image_size=(img_height, img_width),
                                                 batch_size=batch_size)
    class_names = train_data.class_names
    # 返回处理之后的训练集、验证集和类别名
    return train_data, val_data, class_names


# 构建CNN模型--后面统一用类来构建--还可以将多个模型
def model_create(IMG_SHAPE=(224, 224, 3), class_num=10):
    # 搭建模型
    model = Sequential([
        # 对模型做归一化的处理，将0-255之间的数字统一处理到0到1之间
        layers.experimental.preprocessing.Rescaling(1. / 255, input_shape=IMG_SHAPE),

        # 卷积层，该卷积层的输出为32个通道，卷积核的大小是3*3，激活函数为relu(为了非线性)
        layers.Conv2D(32, (3, 3), activation='relu'),
        # 添加池化层，池化的kernel大小是2*2
        layers.MaxPooling2D(2, 2),

        #此处还可以加dropout层

        # 卷积层，输出为64个通道，卷积核大小为3*3，激活函数为relu
        layers.Conv2D(64, (3, 3), activation='relu'),
        # 池化层，最大池化，对2*2的区域进行池化操作
        layers.MaxPooling2D(2, 2),

        # 将二维的输出转化为一维
        layers.Flatten(),

        # The same 128 dense layers, and 10 output layers as in the pre-convolution example:
        layers.Dense(128, activation='relu'),
        # 通过softmax函数将模型输出为类名长度的神经元上，激活函数采用softmax对应概率值
        layers.Dense(class_num, activation='softmax')
    ])
    # 输出模型信息
    model.summary()
    # 指明模型的训练参数，优化器为sgd优化器，损失函数为交叉熵损失函数
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    # 返回模型
    return model


# 展示训练过程的曲线
def show_trainInfo(history):
    # 从history中提取模型训练集和验证集准确率信息和误差信息
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    # 按照上下结构将图画输出
    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.ylabel('Accuracy')
    plt.ylim([min(plt.ylim()), 1])
    plt.title('Training and Validation Accuracy')

    plt.subplot(2, 1, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.ylabel('Cross Entropy')
    plt.title('Training and Validation Loss')
    plt.xlabel('epoch')
    plt.savefig('./TensorFlow/LeNet/results_cnn.png', dpi=100)


def train(epochs):
    # 开始训练，记录开始时间
    begin_time = time()
    # todo 加载数据集， 修改为你的数据集的路径
    train_data, val_data, class_names = data_load("./DataSet/LEDNUM/train_data", "./DataSet/LEDNUM/test_data", 224, 224,
                                                  64)
    print(class_names)
    # 加载模型
    model = model_create(class_num=len(class_names))
    # 指明训练的轮数epoch，开始训练
    history = model.fit(train_data, validation_data=val_data, epochs=epochs)
    # todo 保存模型， 修改为你要保存的模型的名称
    model.save("./TensorFlow/LeNet/cnn_model.h5")
    # 记录结束时间
    end_time = time()
    run_time = end_time - begin_time
    print('训练全过程耗时：', run_time, "s")
    # 绘制模型训练过程图
    show_trainInfo(history)


'''
# 自定义数据集
class MyDataset(datasets):
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
train_data = MyDataset('./LeNet-5/data/LEDNUM/train.txt', transform=pipline_train)
test_data = MyDataset('./LeNet-5/data/LEDNUM/test.txt', transform=pipline_test)

#train_data 和test_data包含多有的训练与测试数据，调用DataLoader批量加载
trainloader = datasets.DataLoader(dataset=train_data, batch_size=32, shuffle=True)
testloader = torch.utils.data.DataLoader(dataset=test_data, batch_size=8, shuffle=False)
'''

if __name__ == '__main__':
    train(epochs=50)
