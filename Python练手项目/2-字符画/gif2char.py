'''
@FileName   :gif2char.py
@Description:
@Date       :2022/10/15 13:38:42
@Author     :daito
@Website    :Https://github.com/zhd5120153951
@Copyright  :daito
@License    :None
@version    :1.0
@Email      :2462491568@qq.com
'''

import cv2
import os
import imageio
from PIL import Image, ImageDraw, ImageFont


# 拆分 gif 将每一帧处理成字符画
def gif2pic(file, chars, isgray, font, scale):
    '''
    file: gif 文件
    ascii_chars: 灰度值对应的字符串
    isgray: 是否黑白
    font: ImageFont 对象
    scale: 缩放比例
    '''
    img = Image.open(file)
    path = os.getcwd()
    if (not os.path.exists(path + "/tmp")):
        os.mkdir(path + "/tmp")
    os.chdir(path + "/tmp")
    #清空tmp目录下内容
    for item in os.listdir(path + "/tmp"):
        os.remove(item)
    try:
        while TRUE:
            current = img.tell()
            name = file.split(".")[0] + "_tmp_" + str(current) + ".png"
            # 保存每一帧图片
            img.save(name)
            # 将每一帧处理为字符画
            img2chars(name, chars, isgray, font, scale)
            # 继续处理下一帧
            img.seek(current + 1)
    except:
        os.chdir(path)


# 将不同的灰度值映射为 ASCII 字符
def get_char(chars, r, g, b):
    length = len(chars)
    gray = int(0.2126 * r + 0.7152 * g + 0.0722 * b)
    return chars[int(gray / (256 / length))]


# 将图片处理成字符画
def img2chars(img, chars, isgray, font, scale):
    sca = scale
    #把图片转为RGB模式
    image = Image.open(img).convert("RGB")
    # 设定处理后的字符画大小
    raw_width = int(image.width * sca)
    raw_height = int(image.height * sca)
    # 获取设定的字体的尺寸
    font_x, font_y = font.getsize(" ")
    # 确定单元的大小
    block_x = int(font_x * sca)
    block_y = int(font_y * sca)
    # 确定长宽各有几个单元
    w = int(raw_width / block_x)
    h = int(raw_height / block_y)
    # 将每个单元缩小为一个像素
    image = image.resize((w, h), Image.NEAREST)
    # txts 和 colors 分别存储对应块的 ASCII 字符和 RGB 值
    txts = []
    colors = []
    for i in range(h):
        line = ""
        lineColor = []
        for j in range(w):
            pixel = image.getpixel((j, i))
            lineColor.append((pixel[0], pixel[1], pixel[2]))
            line += get_char(chars, pixel[0], pixel[1], pixel[2])
        txts.append(line)
        colors.append(lineColor)
    # 创建新画布
    img_txt = Image.new("RGB", (raw_width, raw_height), (255, 255, 255))
    # 创建 ImageDraw 对象以写入 ASCII
    draw = ImageDraw.Draw(img_txt)
    for j in range(len(txts)):
        for i in range(len(txts[0])):
            if isgray:
                draw.text((i * block_x, j * block_y), txts[j][i], (119, 136, 153))
            else:
                draw.text((i * block_x, j * block_y), txts[j][i], colors[j][i])
    img_txt.save(img)


# 读取 tmp 目录下文件合成 gif
def pic2gif(dir_name, out_name, duration):
    path = os.getcwd()
    os.chdir(dir_name)
    dirs = os.listdir()
    images = []
    num = 0
    for d in dirs:
        images.append(imageio.imread(d))
        num += 1
    os.chdir(path)
    imageio.mimsave(out_name + "_char.gif", images, duration=duration)


if __name__ == "__main__":
    chars = list("$@B%8&WM#*oahkbdpqwmZO0QLCJUYXzcvunxrjft/\|()1{}[]?-_+~<>i!lI;:,\"^`'. ")
    fname = 'g2'
    font = ImageFont.truetype('Courier-New.ttf', size=int(6))
    gif2pic(fname + '.gif', chars, False, font, 1)
    pic2gif("tmp", fname, 0.2)
