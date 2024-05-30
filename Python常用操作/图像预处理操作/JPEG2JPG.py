import os
from PIL import Image
from os import listdir
from os.path import isfile, join

# 设置包含JPEG图像的文件夹路径
input_folder = 'D:\\FilePackage\\datasets\\phonecall\\phonecall_3\\phonecall_images'
# 设置保存转换后JPG图像的文件夹路径
output_folder = 'D:\\FilePackage\\datasets\\phonecall\\phonecall_3\\images'


# 检查输出文件夹是否存在，如果不存在则创建
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 遍历文件夹中的所有文件
for file in os.listdir(input_folder):
    if file.endswith('.jpeg') or file.endswith('.png'):
        # 构建原文件和输出文件的完整路径
        input_path = os.path.join(input_folder, file)
        output_path = os.path.join(output_folder, file.replace(
            '.jpeg', '.jpg').replace('.png', '.jpg'))

        # 打开图像并保存为JPG格式
        with Image.open(input_path) as img:
            img = img.convert('RGB')
            img.save(output_path, 'JPEG')

print(
    f"JPEG and PNG images have been converted to JPG and saved in {output_folder}")
