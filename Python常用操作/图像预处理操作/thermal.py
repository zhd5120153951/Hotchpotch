'''
@FileName   :thermal.py
@Description:flir_1_3是第一版数据集,flir_v2是第二版,这个脚本用于查看第一版数据集的图像内容
@Date       :2023/10/07 15:17:20
@Author     :daito
@Website    :Https://github.com/zhd5120153951
@Copyright  :daito
@License    :None
@version    :1.0
@Email      :2462491568@qq.com
'''

from pycocotools.coco import COCO
import matplotlib.pyplot as plt
import cv2

import os
import numpy as np
import random

# 定义变量
dataroot = 'D:\\FilePackage\\BaiduDiskDownload\\Flir\\FLIR_ADAS_1_3' + os.sep

# 加载json文件
jsonfile = dataroot + 'train\\thermal_annotations.json'
coco = COCO(jsonfile)

cls = 'person'
id = coco.getCatIds(cls)[0]
print(f'{cls} 对应的序号为 {id}')

cat = coco.loadCats(id)
print(f'{id} 对应的类别为 {cat}')

#查看特定的图像
ind = random.randint(0, len(coco.imgs))

imInfo = coco.imgs[ind]
annIds = coco.getAnnIds(imgIds=imInfo['id'])
imgfile = dataroot + 'train/' + imInfo['file_name']

print(f'{imInfo} \n对应的 annids 为\n{annIds}\n')

anns = coco.loadAnns(annIds)
if anns:
    print(anns[0])

img = cv2.imread(imgfile)

for ann in anns:
    x, y, w, h = ann['bbox']
    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 1)
    cat = coco.loadCats(ann['category_id'])[0]['name']
    cv2.putText(img, cat, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)

plt.imshow(img)
plt.show()
