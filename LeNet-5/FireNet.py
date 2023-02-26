'''
@FileName   :FireNet.py
@Description:Tensorflow框架移植到Pytorch框架--FireNet
@Date       :2023/02/25 12:22:04
@Author     :daito
@Website    :Https://github.com/zhd5120153951
@Copyright  :daito
@License    :None
@version    :1.0
@Email      :2462491568@qq.com
'''
import os
import cv2
import numpy as np
import random
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import Dataset
from torchvision import transforms
from torch import optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
'''
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, AveragePooling2D
'''
DATADIR = 'Datasets/scrapped/All'
CATEGORIES = ['Fire', 'NoFire']

IMG_SIZE = 64


def create_training_data():
    training_data = []
    for category in CATEGORIES:

        path = os.path.join(DATADIR, category)
        class_num = CATEGORIES.index(category)  # get the classification  (0 or a 1). 0=C 1=O

        for img in tqdm(os.listdir(path)):  # iterate over each image
            try:
                img_array = cv2.imread(os.path.join(path, img))  # convert to array
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))  # resize to normalize data size
                training_data.append([new_array, class_num])  # add this to our training_data
            except Exception as e:  # in the interest in keeping the output clean...
                pass

    return training_data


training_data = create_training_data()

print(len(training_data))
random.shuffle(training_data)
for sample in training_data[:10]:
    print(sample[1])

X = []
Y = []
for features, label in training_data:
    X.append(features)
    Y.append(label)

X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 3)
X = X / 255.0
X.shape[1:]

# # set up image augmentation
# from keras.preprocessing.image import ImageDataGenerator
# datagen = ImageDataGenerator(
#     rotation_range=15,
#     horizontal_flip=True,
#     width_shift_range=0.1,
#     height_shift_range=0.1
#     #zoom_range=0.3
#     )
# datagen.fit(X)


#FireNet模型
class FireNet(nn.Module()):

    def __init__(self) -> None:
        super(FireNet, self).__init__()

        self.conv1 = nn.Conv2d(3, 16, 3)  #in=64*64*3,kernel=3*3*3*16,
        self.relu1 = nn.ReLU()
        self.avgpool1 = nn.AvgPool2d(2, 2)
        self.dropout1 = nn.Dropout(0.5)

        self.conv2 = nn.Conv2d(16, 32, 3)  #in=31*31*16
        self.relu2 = nn.ReLU()
        self.avgpool2 = nn.AvgPool2d(2, 2)
        self.dropout2 = nn.Dropout(0.5)

        self.flatten1 = nn.Flatten()
        self.dense1 = nn.Linear(2304, 256)
        self.dropout3 = nn.Dropout(0.2)
        self.dense2 = nn.Linear(256, 128)
        self.dense3 = nn.Linear(128, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.avgpool1(x)
        x = self.dropout1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.avgpool2(x)
        x = self.dropout2(x)

        x = self.flatten1(x)
        x = self.dense1(x)

        x = self.dropout3(x)
        x = self.dense2(x)
        x = self.dense3(x)
        return x


'''    
import tensorflow as tf
from tensorflow.keras import Sequential, layers, optimizers
#import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# 加载数据集
mnist = tf.keras.datasets.mnist
(trainImage, trainLabel),(testImage, testLabel) = mnist.load_data()
 
for i in [trainImage,trainLabel,testImage,testLabel]:
    print(i.shape)

trainImage = tf.reshape(trainImage,(60000,28,28,1))
testImage = tf.reshape(testImage,(10000,28,28,1))
 
for i in [trainImage,trainLabel,testImage,testLabel]:
    print(i.shape)

# 网络定义
network = Sequential([
    # 卷积层1
    layers.Conv2D(filters=6,kernel_size=(5,5),activation="relu",input_shape=(28,28,1),padding="same"),
    layers.MaxPool2D(pool_size=(2,2),strides=2),
    
    # 卷积层2
    layers.Conv2D(filters=16,kernel_size=(5,5),activation="relu",padding="same"),
    layers.MaxPool2D(pool_size=2,strides=2),
    
    # 卷积层3
    layers.Conv2D(filters=32,kernel_size=(5,5),activation="relu",padding="same"),
    
    layers.Flatten(),
    
    # 全连接层1
    layers.Dense(200,activation="relu"),
    
    # 全连接层2
    layers.Dense(10,activation="softmax")    
])
network.summary()

# 模型训练 训练30个epoch
network.compile(optimizer='adam',loss="sparse_categorical_crossentropy",metrics=["accuracy"])
network.fit(trainImage,trainLabel,epochs=50,validation_split=0.1)

# 模型保存
network.save('./LeNet-5-MNIST-Tensorflow/lenet_mnist.h5')
print('lenet_mnist model saved')
del network
'''