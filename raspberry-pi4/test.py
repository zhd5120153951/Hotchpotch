'''
@FileName   :test.py
@Description:
@Date       :2023/10/19 12:04:01
@Author     :daito
@Website    :Https://github.com/zhd5120153951
@Copyright  :daito
@License    :None
@version    :1.0
@Email      :2462491568@qq.com
'''
import cv2
import torch
import numpy as np
import logging

# 配置基本日志设置
logging.basicConfig(
    level=logging.DEBUG,  # 设置日志级别，可以选择DEBUG, INFO, WARNING, ERROR, CRITICAL
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',  # 设置日志消息的格式
    datefmt='%Y-%m-%d %H:%M:%S',  # 设置日期时间格式
    filename='App.log',  # 指定日志输出到文件
    filemode='a'  # 指定文件写入模式（a表示追加，w表示覆盖）
)

# 创建一个日志记录器
logger = logging.getLogger('my_logger')

print(cv2.__version__)
print(torch.__version__)

 
# cap = cv2.VideoCapture()


if __name__ == "__main__":
    i = 0
    while i<100:
        # get a frame
        # ret, frame = cap.read()
        # if not ret:
            # break
        # show a frame
        # cv2.imshow("capture", frame)
        
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #退出并拍照
        # cv2.imwrite("takephoto2.jpg", frame)
        # print("take Photo Ok")
            # cap.release()
            # cv2.destroyAllWindows()
        
        logger.info("this is {}th log".format(i+1))
        i+=1