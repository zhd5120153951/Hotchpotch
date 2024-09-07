'''
@FileName   :filter.py
@Description:筛选尺寸大于640*480的图像
@Date       :2023/08/07 15:36:22
@Author     :daito
@Website    :Https://github.com/zhd5120153951
@Copyright  :daito
@License    :None
@version    :1.0
@Email      :2462491568@qq.com
'''
import os
import cv2
from PIL import Image


def filter_images(folder_path, width, height, save_path):
    start_number = 1
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)

        #检查文件是否图像文件
        if not os.path.isfile(file_path) or not file_path.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
        try:
            img = cv2.imread(file_path)
            img_height, img_width, _ = img.shape
            # with Image.open(file_path) as img:
            # img.size()

        except (IOError, OSError):
            continue

        #检查图像尺寸是否大于指定尺寸
        if img_width >= width and img_height >= height:
            print(file_path)

            file_ext = os.path.splitext(file_name)[1]  # 获取文件扩展名
            new_filename = f"{'Mouse_'}{start_number:04d}{file_ext}"  # 使用4位序号，并在左侧补0
            save = os.path.join(save_path, new_filename)
            cv2.imwrite(save, img)
            # 更新起始序号
            start_number += 1


if __name__ == '__main__':
    folder_path = '.\Mouse'
    save_path = '.\database\mouse'
    width = 320
    height = 320
    filered_img = filter_images(folder_path, width, height, save_path)
    # 打印筛选后的图像
    # for img in filered_img:
    #     print(img)