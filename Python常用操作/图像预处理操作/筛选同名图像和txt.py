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
import tqdm
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


def copyMatchJpgJsonFile(source_dir, dest_dir):
    if not os.path.exists(source_dir):
        print("原图目录不存在.......")
    # 检查目标目录是否存在，不存在则创建
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    # 获取源目录下的所有文件
    files = os.listdir(source_dir)

    # 创建集合来存储已经匹配的文件名（不含扩展名）--集合的特性--其中的元素不重复
    matched_files = set()

    # 首先找到所有jpg文件并存储其名字（不含扩展名）
    jpg_files = [f for f in files if f.endswith('.jpg')]
    json_files = [f for f in files if f.endswith('.json')]

    # 创建集合，存储jpg文件的名字（不含扩展名）
    jpg_names = set(os.path.splitext(f)[0] for f in jpg_files)

    # 查找与jpg同名的json文件
    for json_file in json_files:
        json_name = os.path.splitext(json_file)[0]
        if json_name in jpg_names:
            # 将同名的jpg和json文件拷贝到目标目录
            matched_files.add(json_name)
            # 构造jpg文件和json文件的完整路径
            jpg_path = os.path.join(source_dir, json_name + '.jpg')
            json_path = os.path.join(source_dir, json_name + '.json')

            # 将它们拷贝到目标目录
            shutil.copy2(jpg_path, dest_dir)
            shutil.copy2(json_path, dest_dir)
            print(f"Copied: {json_name}.jpg and {json_name}.json")

    if not matched_files:
        print("No matching .jpg and .json files found.")
    else:
        print(f"Total {len(matched_files)} pairs of files copied.")


if __name__ == '__main__':
    # img_path = 'D:\\FilePackage\\datasets\\smoke\\images'
    # txt_path = 'D:\\FilePackage\\datasets\\smoke\\labels'
    # # txt_path = 'D:\\FilePackage\\datasets\\SmokeDetection\\train\\labels'
    # txt_save_path = 'D:\\FilePackage\\datasets\\Object Detect\\fire-smoke\\labels'
    # img_save_path = 'D:\\FilePackage\\datasets\\Object Detect\\fire-smoke\\images'
    # new_name_prefix = "smoke_"
    # start_number = 1
    # filered_img = filter_same_name_imgtxt(
    #     img_path, txt_path, txt_save_path, img_save_path)

    # filter_same_name_imgtxt_rename(
    #     img_path, txt_path, new_name_prefix, start_number)

    source_directory = 'E:\\Datasets\\belt\\belt_seg'  # 源文件目录
    destination_directory = 'E:\\Datasets\\belt\\belt_seg_v8'  # 目标文件目录

    copyMatchJpgJsonFile(source_directory, destination_directory)
