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
        preffixe_img = os.path.splitext(img_name)[0]  # img名
        for txt_name in os.listdir(txt_path):
            preffixe_txt = os.path.splitext(txt_name)[0]  # txt名
            if preffixe_img != preffixe_txt:
                continue
            else:
                source_txt_path = os.path.join(txt_path, txt_name)
                source_img_path = os.path.join(img_path, img_name)
                shutil.copy(source_txt_path, save_path)
                shutil.copy(source_img_path, save_path)
            # start_number += 1


if __name__ == '__main__':
    img_path = 'F:\\DataSet\\yolo_format_smoke\\images'
    txt_path = 'F:\\DataSet\\fire-smoke-all\\labels'
    # txt_path = 'D:\\FilePackage\\datasets\\SmokeDetection\\train\\labels'
    save_path = 'F:\\DataSet\\temp_smoke'
    filter_same_name_imgtxt(img_path, txt_path, save_path)
    # 打印筛选后的图像
    # for img in filered_img:
    #     print(img)
