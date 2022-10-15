'''
@FileName   :img2char.py
@Description:
@Date       :2022/10/15 11:36:26
@Author     :daito
@Website    :Https://github.com/zhd5120153951
@Copyright  :daito
@License    :None
@version    :1.0
@Email      :2462491568@qq.com
'''

import cv2
import random
import numpy as np


def img2char(img, k=5):
    if type(img) != np.ndarray:
        img = np.array(img)
    height, width, *_ = img.shape
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_array = np.float32(img_gray.reshape(-1))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TermCriteria_MAX_ITER, 10, 1.0)
    flag = cv2.KMEANS_RANDOM_CENTERS
    #得到label--类别、centeroids矩形中心
    compactness, labels, centeroids = cv2.kmeans(img_array, k, None, criteria, 10, flag)
    centeroids = np.int8(centeroids)
    #label的个数矩形中心随机排列
    centeroids = centeroids.flatten()
    centeroids_sorted = sorted(centeroids)
    #得到不同矩形中心的明暗程度，0--最暗、1--最亮
    centeroids_index = np.array([centeroids_sorted.index(value) for value in centeroids])
    bright = [abs(3 * i - 2 * k) / (3 * k) for i in range(1, 1 + k)]
    bright_bound = bright.index(np.min(bright))
    shadow = [abs((3 * i - k) / (3 * k)) for i in range(1, 1 + k)]
    shadow_bound = shadow.index(np.min(shadow))
    labels = centeroids_index[labels]

    #解析列表
    labels_picked = [labels[rows * width:(rows + 1) * width:2] for rows in range(0, height, 2)]
    #创建长宽==原图三倍的白色背景(画布）8位
    canvas = np.zeros((3 * height, 3 * width, 3), np.uint8)
    canvas.fill(255)

    y = 8
    for rows in labels_picked:
        x = 0
        for cols in rows:
            if cols <= shadow_bound:
                cv2.putText(canvas, str(random.randint(2, 9)), (x, y), cv2.FONT_HERSHEY_PLAIN, 0.45, 1)
            elif cols <= bright_bound:
                cv2.putText(canvas, "-", (x, y), cv2.FONT_HERSHEY_PLAIN, 0.4, 0, 1)
            x += 6
        y += 6
    return canvas


if __name__ == "__main__":
    path = r"static/static.jpg"  #path
    img = cv2.imread(path)
    str_img = img2char(img)
    cv2.imwrite("static/static2char.jpg", str_img)
