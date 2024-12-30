import os
import random
import shutil


def split_and_clean_dataset(images_folder, labels_folder, output_folder, split_ratio=0.8):
    os.makedirs(output_folder, exist_ok=True)

    # 创建主文件夹
    output_images_folder = os.path.join(output_folder, 'images')
    output_labels_folder = os.path.join(output_folder, 'labels')
    os.makedirs(output_images_folder, exist_ok=True)
    os.makedirs(output_labels_folder, exist_ok=True)

    # 获取所有图像文件名
    image_files = [f for f in os.listdir(images_folder) if f.endswith('.jpg')]
    random.shuffle(image_files)

    # 划分训练集和验证集
    split_index = int(split_ratio * len(image_files))
    train_files = image_files[:split_index]
    val_files = image_files[split_index:]

    # 创建训练集和验证集的子文件夹
    train_images_folder = os.path.join(output_images_folder, 'train')
    val_images_folder = os.path.join(output_images_folder, 'val')
    train_labels_folder = os.path.join(output_labels_folder, 'train')
    val_labels_folder = os.path.join(output_labels_folder, 'val')
    os.makedirs(train_images_folder, exist_ok=True)
    os.makedirs(val_images_folder, exist_ok=True)
    os.makedirs(train_labels_folder, exist_ok=True)
    os.makedirs(val_labels_folder, exist_ok=True)

    # 复制图像和标签到相应的文件夹，并检查匹配性
    for file_name in train_files:
        image_path = os.path.join(images_folder, file_name)
        label_name = file_name.replace('.jpg', '.txt')
        label_path = os.path.join(labels_folder, label_name)

        if os.path.exists(label_path):
            shutil.copy(image_path, train_images_folder)
            shutil.copy(label_path, train_labels_folder)
        else:
            print(f"Warning: Label not found for {file_name}. Skipping.")

    for file_name in val_files:
        image_path = os.path.join(images_folder, file_name)
        label_name = file_name.replace('.jpg', '.txt')
        label_path = os.path.join(labels_folder, label_name)

        if os.path.exists(label_path):
            shutil.copy(image_path, val_images_folder)
            shutil.copy(label_path, val_labels_folder)
        else:
            print(f"Warning: Label not found for {file_name}. Skipping.")


# 用法示例：
images_folder_path = 'JPEGImages/'
labels_folder_path = 'output/'
output_folder_path = 'Formost'

split_and_clean_dataset(
    images_folder_path, labels_folder_path, output_folder_path)
