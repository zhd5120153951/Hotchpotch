import xml.etree.ElementTree as ET
import pickle
import os
from os import listdir, getcwd
from os.path import join
import random
from shutil import copyfile

# 类别
CLASSES = ["sleep"]
# 数据集目录
PATH = 'E:\\Datasets\\sleep\\val'
# 训练集占比80% 训练集:验证集=8:2 这里划分数据集 不用改
TRAIN_RATIO = 80


def clear_hidden_files(path):
    dir_list = os.listdir(path)
    for i in dir_list:
        abspath = os.path.join(os.path.abspath(path), i)
        if os.path.isfile(abspath):
            if i.startswith("._"):
                os.remove(abspath)
        else:
            clear_hidden_files(abspath)


def convert(size, box):
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)


def convert_annotation(image_id):
    in_file = open(PATH+'/xml/%s.xml' % image_id, encoding='utf-8')
    out_file = open(PATH+'/txt/%s.txt' %
                    image_id, 'w', encoding='utf-8')
    tree = ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)
    difficult = 0
    for obj in root.iter('object'):
        if obj.find('difficult'):
            difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in CLASSES or int(difficult) == 1:
            # cls = "fire-extinguisher"  # 改为自己的命令
            continue

        cls_id = CLASSES.index(cls)
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text),
             float(xmlbox.find('ymax').text))
        bb = convert((w, h), b)
        out_file.write(str(cls_id) + " " +
                       " ".join([str(a) for a in bb]) + '\n')
    in_file.close()
    out_file.close()


if __name__ == "__main__":

    wd = os.getcwd()

    work_sapce_dir = os.path.join(wd, PATH+"/")
    print(work_sapce_dir)

    annotation_dir = os.path.join(work_sapce_dir, "xml/")
    if not os.path.isdir(annotation_dir):
        os.mkdir(annotation_dir)
    clear_hidden_files(annotation_dir)

    image_dir = os.path.join(work_sapce_dir, "images/")
    if not os.path.isdir(image_dir):
        os.mkdir(image_dir)
    clear_hidden_files(image_dir)

    yolo_labels_dir = os.path.join(work_sapce_dir, "txt/")
    if not os.path.isdir(yolo_labels_dir):
        os.mkdir(yolo_labels_dir)
    clear_hidden_files(yolo_labels_dir)

    list_imgs = os.listdir(image_dir)  # list image files
    print("数据集: %d个" % len(list_imgs))
    for i in range(0, len(list_imgs)):
        path = os.path.join(image_dir, list_imgs[i])
        if os.path.isfile(path):
            image_path = image_dir + list_imgs[i]
            voc_path = list_imgs[i]
            (nameWithoutExtention, extention) = os.path.splitext(
                os.path.basename(image_path))
            (voc_nameWithoutExtention, voc_extention) = os.path.splitext(
                os.path.basename(voc_path))
            annotation_name = nameWithoutExtention + '.xml'
            annotation_path = os.path.join(annotation_dir, annotation_name)
            label_name = nameWithoutExtention + '.txt'
            label_path = os.path.join(yolo_labels_dir, label_name)

        if os.path.exists(annotation_path):
            convert_annotation(nameWithoutExtention)  # convert label
