'''
@FileName   :filterImage.py
@Description:
@Date       :2024/08/28 14:04:11
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


class ImageFilter:
    def __init__(self, src_img_path, target_img_path) -> None:
        self.src_path = src_img_path
        self.target_path = target_img_path

    def create_target_path(self):
        # 如果目标文件夹不存在--创建
        if not os.path.exists(self.target_path):
            os.makedirs(self.target_path)
        else:
            # 文件夹存在--自动新增一个文件夹
            base_name, _ = os.path.split(self.target_path)
            folder_name, _ = os.path.splitext(
                os.path.basename(self.target_path))
            counter = 1
            while True:
                new_folder = f"{base_name}/{folder_name}_{counter}"
                if not os.path.exists(new_folder):
                    self.target_path = new_folder
                    os.makedirs(self.target_path)
                    break
                counter += 1

    def filter_move_image(self):
        # 读取源目录下的图像
        for filename in os.listdir(self.src_path):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(self.src_path, filename)
                try:
                    with Image.open(img_path) as img:
                        width, height = img.size
                        if width > 100 and height > 100:
                            shutil.copy(img_path, os.path.join(
                                self.target_path, filename))
                            print(
                                f"移动文件：{img_path}->{self.target_path}/{filename}")
                except Exception as ex:
                    print(f"无法读取文件：{img_path}-->{ex}")
