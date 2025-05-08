'''
@FileName   :随机划分训练验证集.py
@Description:
@Date       :2024/12/30 09:10:26
@Author     :daito
@Website    :Https://github.com/zhd5120153951
@Copyright  :daito
@License    :None
@version    :1.0
@Email      :2462491568@qq.com
'''
import os
import random
import shutil

# 输入文件夹路径和划分比例
folder_path = "E:\\Datasets\\sleep\\labeled-data\\sleep-v3"
train_ratio = 0.8

# 检查文件夹是否存在
if not os.path.exists(folder_path):
    print("文件夹不存在！")
    exit()

# 获取所有jpg和txt文件
jpg_files = [file for file in os.listdir(folder_path) if file.endswith(".jpg")]
txt_files = [file for file in os.listdir(folder_path) if file.endswith(".txt")]

# 检查文件数量是否相等
if len(jpg_files) != len(txt_files):
    print("图片和标签数量不匹配！")
    exit()

# 打乱文件顺序
random.shuffle(jpg_files)

# 划分训练集和验证集
train_size = int(len(jpg_files) * train_ratio)
train_jpg = jpg_files[:train_size]
train_txt = [file.replace(".jpg", ".txt") for file in train_jpg]
val_jpg = jpg_files[train_size:]
val_txt = [file.replace(".jpg", ".txt") for file in val_jpg]

# 创建文件夹和子文件夹
img_train_folder = os.path.join(folder_path, "images/train")
img_val_folder = os.path.join(folder_path, "images/val")
txt_train_folder = os.path.join(folder_path, "labels/train")
txt_val_folder = os.path.join(folder_path, "labels/val")
if not os.path.exists(img_train_folder):
    os.makedirs(img_train_folder)
if not os.path.exists(img_val_folder):
    os.makedirs(img_val_folder)
if not os.path.exists(txt_train_folder):
    os.makedirs(txt_train_folder)
if not os.path.exists(txt_val_folder):
    os.makedirs(txt_val_folder)

# 复制文件到目标文件夹
# for file in train_jpg:
#     shutil.copy(os.path.join(folder_path, file), img_train_folder)

# for file in train_txt:
#     shutil.copy(os.path.join(folder_path, file), txt_train_folder)
# for file in val_jpg:
#     shutil.copy(os.path.join(folder_path, file), img_val_folder)
# for file in val_txt:
#     shutil.copy(os.path.join(folder_path, file), txt_val_folder)

# 移动文件到目标文件夹
for file in train_jpg:
    shutil.move(os.path.join(folder_path, file), img_train_folder)
for file in train_txt:
    shutil.move(os.path.join(folder_path, file), txt_train_folder)
for file in val_jpg:
    shutil.move(os.path.join(folder_path, file), img_val_folder)
for file in val_txt:
    shutil.move(os.path.join(folder_path, file), txt_val_folder)

print("处理完成！")
