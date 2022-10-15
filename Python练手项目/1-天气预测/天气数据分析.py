import cv2
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdate
from dateutil import parser

#不同城市的气象数据导入
dataFerrara = pd.read_csv("WeatherData/ferrara_270615.csv")
dataMilano = pd.read_csv("WeatherData/milano_270615.csv")
dataMantova = pd.read_csv("WeatherData/mantova_270615.csv")
dataRavenna = pd.read_csv("WeatherData/ravenna_270615.csv")
dataTorino = pd.read_csv("WeatherData/torino_270615.csv")
dataAsti = pd.read_csv("WeatherData/asti_270615.csv")
dataBologna = pd.read_csv("WeatherData/bologna_270615.csv")
dataPiacenza = pd.read_csv("WeatherData/piacenza_270615.csv")
dataCesena = pd.read_csv("WeatherData/cesena_270615.csv")
dataFaenza = pd.read_csv("WeatherData/faenza_270615.csv")

#从csv中取出需要分析的数据--温度和日期
temperature = dataMilano["temp"]
date = dataMilano["day"]

#转换日期格式--datetime格式--格式化函数paeser类中的parser函数
dateConvert = [parser.parse(i) for i in date]

# 调用 subplot函数绘制坐标轴, fig 是图像对象，ax 是坐标轴对象
fig, ax = plt.subplots()

# 调整x轴坐标刻度，使其旋转70度，方便查看--为了不拥挤
plt.xticks(rotation=70)

# 设定时间的格式--这里是指时分秒格式
timeConvert = mdate.DateFormatter("%H:%M")

# 设定X轴显示的变量
ax.xaxis.set_major_formatter(timeConvert)

# 画出图像,X轴数据-日期,Y轴数据-温度,r代表的是-红色线条绘制
ax.plot(dateConvert, temperature, "r")
