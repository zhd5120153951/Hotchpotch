import os
import random
import argparse

parser = argparse.ArgumentParser()
#xml文件的路径,根据自己的修改，xml一般在Annotations下，这里没有直接生成yolo_txt格式，而是通过xml来转
parser.add_argument('--xml_path',default='Annotations',type=str,help='input xml')
#数据集划分，路径可以选择自己数据集下的ImageSets/Main
parser.add_argument('--txt_path',default='labels')