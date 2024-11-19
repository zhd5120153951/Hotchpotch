'''
@FileName   :录频录音.py
@Description:
@Date       :2024/11/19 18:54:43
@Author     :daito
@Website    :Https://github.com/zhd5120153951
@Copyright  :daito
@License    :None
@version    :1.0
@Email      :2462491568@qq.com
'''
import tkinter as tk
import tkinter.messagebox as messagebox
import pyautogui
import cv2
import numpy as np
from datetime import datetime
import threading
import pyaudio
import wave
import os
from moviepy.editor import VideoFileClip, AudioFileClip

# 隐藏命令行
import win32console
import win32gui
win = win32console.GetConsoleWindow()
win32gui.ShowWindow(win, 0)


class Popup(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("设置")
        self.geometry("300x250")
        self.configure(bg="#333333")
        self.create_widgets()
        self.recording_area = None  # 初始化录制区域
        self.record_microphone = True  # 初始化打开麦克风

    def create_widgets(self):
        self.recording = False
        self.stop_event = threading.Event()  # 使用 Event 对象

        self.rec_button = tk.Button(self, text="开始录制", command=self.toggle_recording,
                                    bg="#333333", fg="white", borderwidth=2, relief="solid")
        self.rec_button.pack(pady=10)

        self.stop_button = tk.Button(self, text="停止录制", command=self.stop_recording,
                                     bg="#333333", fg="white", borderwidth=2, relief="solid", state=tk.DISABLED)
        self.stop_button.pack(pady=5)

        # 区域选择按钮
        self.select_area_button = tk.Button(
            self, text="选择录制区域", command=self.select_area, bg="#333333", fg="white", borderwidth=2, relief="solid")
        self.select_area_button.pack(pady=10)

        # 扬声器录制选项
        self.mic_checkbox = tk.Checkbutton(
            self, text="不录制扬声器系统声音", command=self.toggle_microphone, bg="#333333", fg="white", selectcolor="black")
        self.mic_checkbox.pack(pady=10)

    def toggle_microphone(self):
        # 切换麦克风录制的状态
        self.record_microphone = not self.record_microphone

    def select_area(self):
        # 打开一个全屏窗口，允许用户通过鼠标拖动选择区域
        self.selection_window = tk.Toplevel(self)
        self.selection_window.attributes('-fullscreen', True)
        self.selection_window.attributes('-alpha', 0.3)  # 半透明
        self.selection_window.config(bg="gray")

        self.canvas = tk.Canvas(self.selection_window,
                                cursor="cross", bg="gray")
        self.canvas.pack(fill="both", expand=True)

        self.rect_id = None
        self.start_x = None
        self.start_y = None

        # 绑定鼠标事件
        self.canvas.bind("<ButtonPress-1>", self.on_mouse_down)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_up)

    def on_mouse_down(self, event):
        # 记录起点
        self.start_x = event.x
        self.start_y = event.y
        self.rect_id = self.canvas.create_rectangle(
            self.start_x, self.start_y, self.start_x, self.start_y, outline='red', width=2)

    def on_mouse_drag(self, event):
        # 动态更新矩形
        self.canvas.coords(self.rect_id, self.start_x,
                           self.start_y, event.x, event.y)

    def on_mouse_up(self, event):
        # 记录终点并计算区域
        end_x = event.x
        end_y = event.y
        self.recording_area = (self.start_x, self.start_y,
                               end_x - self.start_x, end_y - self.start_y)
        self.selection_window.destroy()
        messagebox.showinfo("信息", f"选择的录制区域为：{self.recording_area}")

    def toggle_recording(self):
        if not self.recording:
            self.start_recording()
        else:
            self.stop_recording()

    def start_recording(self):
        self.recording = True
        self.stop_event.clear()  # 清除事件状态
        frame_rate = 30
        timestamp = self.get_current_timestamp()
        self.video_filename = f"recorded_video_{timestamp}.mp4"
        self.audio_filename = f"recorded_audio_{timestamp}.wav"

        if self.recording_area is None:
            screen_width, screen_height = pyautogui.size()
            self.recording_area = (0, 0, screen_width, screen_height)  # 默认全屏

        codec = cv2.VideoWriter_fourcc(*"mp4v")
        self.video_out = cv2.VideoWriter(
            self.video_filename, codec, frame_rate, (self.recording_area[2], self.recording_area[3]))

        # 判断是否需要录制麦克风
        if self.record_microphone:
            # 声音录制线程
            self.audio_thread = threading.Thread(
                target=self.record_audio, args=(self.audio_filename,))
            self.audio_thread.start()

        # 启动录制线程
        self.recording_thread = threading.Thread(target=self.record_screen)
        self.recording_thread.start()

        self.rec_button.config(text="正在录制", state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)

    def record_screen(self):
        while not self.stop_event.is_set():  # 检查事件状态
            img = pyautogui.screenshot(region=self.recording_area)  # 录制自定义区域
            frame = np.array(img)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            self.video_out.write(frame)

        self.video_out.release()
        print("视频录制结束")

    def record_audio(self, filename):
        chunk = 1024
        sample_format = pyaudio.paInt16
        channels = 2
        fs = 44100  # 采样频率

        p = pyaudio.PyAudio()

        # 打开麦克风输入流
        stream = p.open(format=sample_format,
                        channels=channels,
                        rate=fs,
                        input=True,
                        frames_per_buffer=chunk)

        frames = []

        while not self.stop_event.is_set():  # 检查事件状态
            data = stream.read(chunk)
            frames.append(data)

        stream.stop_stream()
        stream.close()
        p.terminate()

        # 保存音频到 wav 文件
        with wave.open(filename, 'wb') as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(p.get_sample_size(sample_format))
            wf.setframerate(fs)
            wf.writeframes(b''.join(frames))

        print("录音结束")

    def stop_recording(self):
        self.recording = False
        self.stop_event.set()  # 设定事件，结束录制

        if self.record_microphone:
            self.audio_thread.join()  # 等待音频线程结束

        if os.path.isfile('merged_video.mp4'):
            messagebox.showinfo("提示", "合并视频文件已保存至当前目录文件夹，想重新录制请删除原文件后重试！")
            # 更新按钮状态
            self.rec_button.config(text="请删除原文件后再开始录制", state=tk.NORMAL)
            self.stop_button.config(state=tk.DISABLED)
        else:
            # 合成音视频
            self.merge_audio_video()

    def merge_audio_video(self):
        merge_file = 'merged_video.mp4'

        video_clip = VideoFileClip(self.video_filename)

        if self.record_microphone:
            audio_clip = AudioFileClip(self.audio_filename)
            final_clip = video_clip.set_audio(audio_clip)
        else:
            final_clip = video_clip  # 如果没有录音，只保存视频

        final_clip.write_videofile(
            merge_file, codec='libx264', audio_codec="aac")

        # 删除录制的视频文件和音频文件
        os.remove(self.video_filename)
        if self.record_microphone:
            os.remove(self.audio_filename)

        print("音视频合成完成，存储为", merge_file)
        # 更新按钮状态
        self.rec_button.config(text="开始录制", state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)

    def get_current_timestamp(self):
        return datetime.now().strftime("%Y%m%d_%H%M%S")


if __name__ == "__main__":
    popup = Popup()
    popup.mainloop()
