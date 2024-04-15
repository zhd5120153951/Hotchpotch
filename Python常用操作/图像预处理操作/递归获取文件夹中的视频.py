import os
import shutil

# 定义一个函数来检查文件是否是视频文件


def is_video_file(filename):
    video_extensions = ('.mp4', '.avi', '.mkv', '.mov', '.flv', '.wmv')
    return filename.lower().endswith(video_extensions)

# 定义一个递归函数来移动视频文件


def move_videos(directory, target_directory):
    # 检查目标目录是否存在，如果不存在则创建
    if not os.path.exists(target_directory):
        os.makedirs(target_directory)

    # 遍历目录中的文件和子目录
    for filename in os.listdir(directory):
        # 获取文件的完整路径
        file_path = os.path.join(directory, filename)

        # 检查当前文件是否是文件夹
        if os.path.isdir(file_path):
            # 递归调用函数，移动子目录中的视频文件
            move_videos(file_path, target_directory)
        elif is_video_file(filename):
            # 获取源文件和目标文件的完整路径
            source_file_path = file_path
            target_file_path = os.path.join(target_directory, filename)

            # 移动视频文件
            shutil.move(source_file_path, target_file_path)
            print(f"视频文件 {filename} 已移动到 {target_file_path}")


if __name__ == "__main__":
    # 使用方法：传入要移动视频文件的根目录和目标目录
    dist_path = "C:\\Users\\Zengh\\Downloads\\zhongke_video"
    source_path = "C:\\Users\\Zengh\\Downloads\\video"
    move_videos(source_path, dist_path)
