#测试SE5的服务自启动--开机再重启--针对推理python进程

import cv2
import os
import json
import logging 
import argparse
import numpy as np

# 配置基本日志设置
logging.basicConfig(
    level=logging.DEBUG,  # 设置日志级别
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',  # 设置日志消息的格式
    datefmt='%Y-%m-%d %H:%M:%S',  # 设置日期时间格式
    filename='myapp.log',   # 指定日志输出到文件
    filemode='a'            # 指定文件写入模式（a表示追加，w表示覆盖）
)

# 创建一个日志记录器
logger = logging.getLogger('my_logger')

def auto_start():
    read_json_config()
    pass

def read_json_config(json_path):
    pass


if __name__ == "__main__":    
    auto_start()    #在这里服务起来的进程--因该用线程开启其他功能
