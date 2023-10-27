'''
@FileName   :filter.py
@Description:筛选尺寸大于640*480的图像
@Date       :2022/08/07 15:36:22
@Author     :daito
@Website    :Https://github.com/zhd5120153951
@Copyright  :daito
@License    :None
@version    :1.0
@Email      :2462491568@qq.com
'''
import os
import cv2
import shutil
from PIL import Image


def filter_same_name_imgtxt(img_path, txt_path, save_path):
    # start_number = 1
    for img_name in os.listdir(img_path):
        for txt_name in os.listdir(txt_path):
            if img_name != txt_name:
                continue
            else:
                source_txt_path = os.path.join(txt_path, txt_name)
                shutil.copy(source_txt_path, save_path)
            # start_number += 1


if __name__ == '__main__':
    img_path = 'D:\\FilePackage\\datasets\\gas'
    txt_path = 'D:\\FilePackage\\datasets\\SmokeDetection\\train\\labels'
    save_path = 'D:\\FilePackage\\datasets\\gas_txt\\'
    filered_img = filter_same_name_imgtxt(img_path, txt_path, save_path)
    # 打印筛选后的图像
    # for img in filered_img:
    #     print(img)