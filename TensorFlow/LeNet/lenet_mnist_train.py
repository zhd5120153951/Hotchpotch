'''
@FileName   :lenet_mnist_train.py
@Description:FireNet模型训练大尺度图像
@Date       :2023/03/04 09:24:41
@Author     :daito
@Website    :Https://github.com/zhd5120153951
@Copyright  :daito
@License    :None
@version    :1.0
@Email      :2462491568@qq.com
'''
import tensorflow as tf
from tensorflow.keras import Sequential, layers, optimizers
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, AveragePooling2D, MaxPool2D
#import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# 加载数据集
mnist = tf.keras.datasets.mnist
(trainImage, trainLabel), (testImage, testLabel) = mnist.load_data()

for i in [trainImage, trainLabel, testImage, testLabel]:
    print(i.shape)

trainImage = tf.reshape(trainImage, (60000, 28, 28, 1))
testImage = tf.reshape(testImage, (10000, 28, 28, 1))

for i in [trainImage, trainLabel, testImage, testLabel]:
    print(i.shape)

model = Sequential()

model.add(Conv2D(filters=16, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1), padding='same'))
model.add(MaxPool2D(pool_size=(2, 2), strides=2))
model.add(Dropout(0.5))

model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2), strides=2))
model.add(Dropout(0.5))

model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
#model.add(AveragePooling2D())
#model.add(Dropout(0.5))

model.add(Flatten())

model.add(Dense(256, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(128, activation='relu'))

model.add(Dense(10, activation='softmax'))

model.summary()
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(trainImage, trainLabel, epochs=50, validation_split=0.1)

model.save('./TensorFlow/LeNet/firenet_mnist.h5')
print('firenet_mnist model saved')
del model
'''
# LeNet-5
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
