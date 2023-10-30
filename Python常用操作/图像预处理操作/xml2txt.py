import xml.etree.ElementTree as ET
import os
import glob

classes = ["fire", "nofire"]


def convert(size, box):
    dw = 1. / (size[0])
    dh = 1. / (size[1])
    x1 = box[0] * dw
    x2 = box[1] * dw
    x3 = box[1] * dw
    x4 = box[0] * dw

    y1 = box[2] * dh
    y2 = box[2] * dh
    y3 = box[3] * dh
    y4 = box[3] * dh

    return x1, x2, x3, x4, y1, y2, y3, y4


def convert2(size, box):
    dw = 1. / (size[0])
    dh = 1. / (size[1])

    w = box[1] - box[0]
    h = box[3] - box[2]
    x = box[0] + w / 2
    y = box[2] + h / 2

    x = x * dw
    y = y * dh
    w = w * dw
    h = h * dh

    return x, y, w, h


#处理单个xml文件
def convert_annotation(label_xml, label_txt_path, label_xml_path):
    basename = os.path.splitext(label_xml)[0]  #父目录
    print(basename)

    in_file = open(os.path.join(label_xml_path, label_xml), encoding="utf-8")  #图像对应的xml地址
    out_file = open(os.path.join(label_txt_path, "{}.txt".format(basename)), "w")
    tree = ET.parse(in_file)
    root = tree.getroot()
    size = root.find("size")

    w = int(size.find("width").text)
    h = int(size.find("height").text)

    #在一个xml中每一个Object的迭代转换
    b = root.iter("object")
    for obj in root.iter("object"):
        # iter()迭代器方法可以递归遍历元素/树的全部子节点
        difficult = obj.find("difficult").text
        cls = obj.find("name").text
        # 如果是训练标签中的类别不在程序预设值，difficult=1，跳过次Object
        if cls not in classes or int(difficult) == 1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find("bndbox")

        b = (float(xmlbox.find("xmin").text), float(xmlbox.find("xmax").text), float(xmlbox.find("ymin").text),
             float(xmlbox.find("ymax").text))

        bb = convert2((w, h), b)

        out_file.write(str(cls_id) + "" + "".join([str(a) for a in bb]) + "\n")


def generate_txt(label_xml_path, label_txt_path):
    if not os.path.exists(label_txt_path):  #不存在文件夹时
        os.makedirs(label_txt_path)
    label_xmls = os.listdir(label_xml_path)
    for label_xml in label_xmls:
        convert_annotation(label_xml, label_txt_path, label_xml_path)


if __name__ == "__main__":
    generate_txt(label_xml_path="..\Annotations", label_txt_path="..\txt")
