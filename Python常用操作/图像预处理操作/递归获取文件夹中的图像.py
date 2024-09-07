import os
import shutil

# 指定源文件夹路径
source_folder = 'F:\\BaiduNetdiskDownload\\pic\\pic'

# 指定目标文件夹路径
target_folder = 'F:\\BaiduNetdiskDownload\\pic\\all_pic'

# 遍历源文件夹中的所有子文件夹
for foldername in os.listdir(source_folder):
    # 获取子文件夹的完整路径
    source_folder_path = os.path.join(source_folder, foldername)
    # 判断子文件夹是否存在
    if os.path.exists(source_folder_path):
        # 遍历子文件夹中的所有文件
        for ch_filename in os.listdir(source_folder_path):
            ch_source_folder_path = os.path.join(
                source_folder_path, ch_filename)
            if os.path.exists(ch_source_folder_path):
                for filename in os.listdir(ch_source_folder_path):
                    # 判断文件是否为jpg格式
                    if filename.endswith('.jpg'):
                        # 获取文件的完整路径
                        file_path = os.path.join(
                            ch_source_folder_path, filename)
                        # 将文件复制到目标文件夹中
                        shutil.copy2(file_path, target_folder)
