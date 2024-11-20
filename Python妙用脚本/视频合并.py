'''
@FileName   :视频合并.py
@Description:
@Date       :2024/11/20 11:46:42
@Author     :daito
@Website    :Https://github.com/zhd5120153951
@Copyright  :daito
@License    :None
@version    :1.0
@Email      :2462491568@qq.com
'''
import tkinter as tk
from tkinter import filedialog, messagebox
from moviepy.editor import VideoFileClip, concatenate_videoclips


# 定义合并视频的函数
def merge_videos_with_transition(video1_path, video2_path, output_path):
    try:
        # 加载两个视频
        video1 = VideoFileClip(video1_path).fadeout(1)  # 1秒淡出
        video2 = VideoFileClip(video2_path).fadein(1)  # 1秒淡入

        # 合并两个视频
        final_video = concatenate_videoclips(
            [video1, video2], method="compose")

        # 输出合并后的视频
        final_video.write_videofile(
            output_path, codec="libx264", audio_codec="aac")
        messagebox.showinfo("成功", "视频合并成功！")
    except Exception as e:
        messagebox.showerror("错误", f"发生错误: {e}")


# 定义按钮点击事件，选择视频1文件
def select_video1():
    video1_path.set(filedialog.askopenfilename(
        filetypes=[("MP4 files", "*.mp4")]))


# 定义按钮点击事件，选择视频2文件
def select_video2():
    video2_path.set(filedialog.askopenfilename(
        filetypes=[("MP4 files", "*.mp4")]))


# 定义按钮点击事件，选择输出路径
def select_output():
    output_path.set(filedialog.asksaveasfilename(
        defaultextension=".mp4", filetypes=[("MP4 files", "*.mp4")]))


if __name__ == '__main__':
    # 创建主窗口
    root = tk.Tk()
    root.title("视频合并工具")

    # 设置窗口大小
    root.geometry("400x300")

    # 创建界面控件
    video1_path = tk.StringVar()
    video2_path = tk.StringVar()
    output_path = tk.StringVar()

    # 视频1选择框
    tk.Label(root, text="选择第一个视频").pack(pady=10)
    tk.Button(root, text="选择视频1", command=select_video1).pack(pady=5)
    tk.Entry(root, textvariable=video1_path, width=50).pack(pady=5)

    # 视频2选择框
    tk.Label(root, text="选择第二个视频").pack(pady=10)
    tk.Button(root, text="选择视频2", command=select_video2).pack(pady=5)
    tk.Entry(root, textvariable=video2_path, width=50).pack(pady=5)

    # 输出路径选择框
    tk.Label(root, text="选择输出文件路径").pack(pady=10)
    tk.Button(root, text="选择输出路径", command=select_output).pack(pady=5)
    tk.Entry(root, textvariable=output_path, width=50).pack(pady=5)

    # 合并按钮
    tk.Button(root, text="合并视频",
              command=lambda: merge_videos_with_transition(video1_path.get(), video2_path.get(), output_path.get())).pack(
        pady=20)
    # 启动 GUI 主循环
    root.mainloop()
