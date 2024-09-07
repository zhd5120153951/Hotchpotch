import cv2
import pyautogui
import numpy as np

# 获取屏幕尺寸
screen_width, screen_height = pyautogui.size()

# 设置输出视频的帧率、编码器、分辨率等参数
frame_rate = 20
output_filename = 'screen_record.mp4'
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
output_video = cv2.VideoWriter(
    output_filename, fourcc, frame_rate, (screen_width, screen_height))

# 开始录屏
try:
    while True:
        # 获取屏幕截图=
        screenshot = pyautogui.screenshot()

        # 将截图转换成视频帧
        frame = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)

        # 写入视频文件
        output_video.write(frame)

        # 按下esc键停止录屏
        if cv2.waitKey(1) == 27:
            break

except KeyboardInterrupt:
    pass

finally:
    # 释放资源
    cv2.destroyAllWindows()
    output_video.release()
