'''
@FileName   :test.py
@Description:
@Date       :2022/10/09 15:01:00
@Author     :daito
@Website    :Https://github.com/zhd5120153951
@Copyright  :daito
@License    :None
@version    :1.0
@Email      :2462491568@qq.com
'''
import cv2 as cv
import numpy as np
import tensorflow as tf
import argparse
import sys
import os.path
import random
import os
import glob
import operator

image_size = 256  # 128
num_channels = 3
images = []

outputFile = sys.argv[2]

# Opening frames
cap = cv.VideoCapture(sys.argv[1])

vid_writer = cv.VideoWriter(outputFile, cv.VideoWriter_fourcc('M', 'J', 'P', 'G'), 15,
                            (round(cap.get(cv.CAP_PROP_FRAME_WIDTH)), round(cap.get(cv.CAP_PROP_FRAME_HEIGHT))))

width = int(round(cap.get(cv.CAP_PROP_FRAME_WIDTH)))
height = int(round(cap.get(cv.CAP_PROP_FRAME_HEIGHT)))

newHeight = int(round(height / 2))

# Restoring the model
sess = tf.compat.v1.Session()
saver = tf.compat.v1.train.import_meta_graph('roadsurface-model.meta')
saver.restore(sess, tf.compat.v1.train.latest_checkpoint('./'))

# Acessing the graph
graph = tf.get_default_graph()

#
y_pred = graph.get_tensor_by_name("y_pred:0")

#
x = graph.get_tensor_by_name("x:0")
y_true = graph.get_tensor_by_name("y_true:0")
y_test_images = np.zeros(
    (1, len(os.listdir('E:\\Source\\Github\\datasets\\roadsurface-classification\\training_data'))))

while cv.waitKey(1) < 0:

    hasFrame, images = cap.read()

    finalimg = images

    if not hasFrame:
        print("Classification done!")
        print("Results saved as: ", outputFile)
        cv.waitKey(3000)
        break

    images = images[newHeight - 5:height - 50, 0:width]
    images = cv.resize(images, (image_size, image_size), 0, 0, cv.INTER_LINEAR)
    images = np.array(images, dtype=np.uint8)
    images = images.astype('float32')
    images = np.multiply(images, 1.0 / 255.0)

    x_batch = images.reshape(1, image_size, image_size, num_channels)

    #
    feed_dict_testing = {x: x_batch, y_true: y_test_images}
    result = sess.run(y_pred, feed_dict=feed_dict_testing)

    outputs = [result[0, 0], result[0, 1], result[0, 2]]

    value = max(outputs)
    index = np.argmax(outputs)

    if index == 0:
        label = 'Asphalt'
        prob = str("{0:.2f}".format(value))
        color = (0, 0, 0)
    elif index == 1:
        label = 'Paved'
        prob = str("{0:.2f}".format(value))
        color = (153, 102, 102)
    elif index == 2:
        label = 'Unpaved'
        prob = str("{0:.2f}".format(value))
        color = (0, 153, 255)

    cv.rectangle(finalimg, (0, 0), (145, 40), (255, 255, 255), cv.FILLED)
    cv.putText(finalimg, 'Class: ', (5, 15),
               cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    cv.putText(finalimg, label, (70, 15),
               cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    cv.putText(finalimg, prob, (5, 35),
               cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    vid_writer.write(finalimg.astype(np.uint8))

sess.close()
