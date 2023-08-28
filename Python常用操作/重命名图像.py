import os


#重命名文件名
def rename_files(folder_path, new_name_prefix):
    """
    将文件夹中的文件按照给定前缀和序号进行批量重命名。

    参数:
        - folder_path (str): 文件夹路径
        - new_name_prefix (str): 新文件名的前缀

    返回:
        - None
    """
    start_number = 2498
    # 遍历文件夹中的文件
    for filename in os.listdir(folder_path):
        print(filename)
        # 构建新文件名
        file_ext = os.path.splitext(filename)[1]  # 获取文件扩展名
        new_filename = f"{new_name_prefix}{start_number:05d}{file_ext}"  # 使用3位序号，并在左侧补0
        # 构建文件的完整路径
        old_filepath = os.path.join(folder_path, filename)
        new_filepath = os.path.join(folder_path, new_filename)
        # 重命名文件
        os.rename(old_filepath, new_filepath)
        # print(f"将文件 '{filename}' 重命名为 '{new_filename}'")
        # 更新起始序号
        start_number += 1

    print("rename file successfully")


#重命名后缀为jpg文件
def rename_Img(path_orig, path_dst):
    imgList = os.listdir(path_orig)
    for img in imgList:
        if img.endswith(".jpg"):
            name = img.split(".", 3)[0] + "." + img.split(".", 3)[1]
            src = os.path.join(os.path.abspath(path_orig), img)
            dst = os.path.join(os.path.abspath(path_dst), name + ".jpg")
            try:
                os.rename(src, dst)
            except:
                continue


if __name__ == "__main__":
    folder_path = 'D:\\FilePackage\\datasets\\Object Detect\\fire\\images\\val'
    new_name_prefix = 'fire_'
    rename_files(folder_path, new_name_prefix)
