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
        self.dense1 = 
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.maxpool2(x)
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        output = F.log_softmax(x, dim=1)
        return output



model.add(Flatten())
model.add(Dense(units=256, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(units=128, activation='relu'))

model.add(Dense(units=2, activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model.fit(X, Y, batch_size=32, epochs=100, validation_split=0.3)
# model.fit_generator(datagen.flow(X, Y, batch_size=32),
#                     epochs=100,
#                     verbose=1)

model.save('TrainedModels/Fire-64x64-color-v7.1-soft.h5')

from matplotlib import pyplot as plt

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

import tensorflow as tf
from tensorflow.keras.utils import plot_model

model = tf.keras.models.load_model('TrainedModels/Fire-64x64-color-v7-soft.h5')

# model.fit_generator(datagen.flow(X, Y, batch_size=32),
#                     epochs=100,
#                   verbose=1)

# plot_model(model, to_file='model_small.svg', show_layer_names=False, show_shapes=True)"
