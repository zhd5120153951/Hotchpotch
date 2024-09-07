'''
@FileName   :不规则尺寸2规则尺寸.py
@Description:把尺寸不同的图变换为尺寸相同的图,周围都是纯白的--放大
@Date       :2023/11/16 09:58:16
@Author     :daito
@Website    :Https://github.com/zhd5120153951
@Copyright  :daito
@License    :None
@version    :1.0
@Email      :2462491568@qq.com
'''

import cv2
import os

folder = "./Python常用操作/图像预处理操作/images"

new_folder = f"{folder}_new"

if not os.path.exists(new_folder):
    os.mkdir(new_folder)
for filename in os.listdir(folder):
    if '.jpg' not in filename.lower():
        continue
    filepath = f"{folder}/{filename}"
    img = cv2.imread(filepath)

    h, w = img.shape[:2]  # 不获取通道
    px, py = 0, 0
    if h > w:  # 行数大于列数---高大于宽
        px = (h-w)//2
    elif h < w:  # 反过来
        py = (w-h)//2
    newImg = cv2.copyMakeBorder(
        img, py, py, px, px, borderType=cv2.BORDER_CONSTANT, value=[255, 255, 255])
    cv2.imwrite(f"{new_folder}/{filename}", newImg)
