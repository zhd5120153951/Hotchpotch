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
import glob
from PIL import Image


def filter_same_name_imgtxt(img_path, txt_path, txt_save_path, img_save_path):
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
                shutil.copy(source_txt_path, txt_save_path)
                shutil.copy(source_img_path, img_save_path)

            # start_number += 1


def filter_same_name_imgtxt_rename(img_path, txt_path, new_name_prefix, start_number):
    for img_name in os.listdir(img_path):
        preffixe_img = os.path.splitext(img_name)[0]  # img名
        for txt_name in os.listdir(txt_path):
            preffixe_txt = os.path.splitext(txt_name)[0]  # txt名
            if preffixe_img != preffixe_txt:
                continue
            else:
                # 构建新文件名
                img_file_ext = os.path.splitext(img_name)[1]  # 获取图像文件扩展名--后缀
                txt_file_ext = os.path.splitext(txt_name)[1]  # 获取标签文件扩展名--后缀

                # 使用3位序号，并在左侧补0
                new_imgfilename = f"{new_name_prefix}{start_number:05d}{img_file_ext}"
                new_txtfilename = f"{new_name_prefix}{start_number:05d}{txt_file_ext}"

                # 构建文件的完整路径
                old_imgfilepath = os.path.join(img_path, img_name)
                old_txtfilepath = os.path.join(txt_path, txt_name)

                new_imgfilepath = os.path.join(img_path, new_imgfilename)
                new_txtfilepath = os.path.join(txt_path, new_txtfilename)

                # 重命名文件
                os.rename(old_imgfilepath, new_imgfilepath)
                os.rename(old_txtfilepath, new_txtfilepath)

                # print(f"将文件 '{filename}' 重命名为 '{new_filename}'")
                # 更新起始序号
                start_number += 1
def 
if __name__ == '__main__':
    img_path = 'D:\\FilePackage\\datasets\\smoke\\images'
    txt_path = 'D:\\FilePackage\\datasets\\smoke\\labels'
    # txt_path = 'D:\\FilePackage\\datasets\\SmokeDetection\\train\\labels'
    txt_save_path = 'D:\\FilePackage\\datasets\\Object Detect\\fire-smoke\\labels'
    img_save_path = 'D:\\FilePackage\\datasets\\Object Detect\\fire-smoke\\images'
    new_name_prefix = "smoke_"
    start_number = 1
    # filered_img = filter_same_name_imgtxt(
    #     img_path, txt_path, txt_save_path, img_save_path)

    filter_same_name_imgtxt_rename(
        img_path, txt_path, new_name_prefix, start_number)
