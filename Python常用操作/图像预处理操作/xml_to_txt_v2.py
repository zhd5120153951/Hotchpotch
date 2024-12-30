import os
import xml.etree.ElementTree as ET


def convert_voc_to_yolo(xml_path, classes, output_folder):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    image_filename = root.find('filename').text
    image_width = float(root.find('size/width').text)
    image_height = float(root.find('size/height').text)

    yolo_lines = []

    for obj in root.findall('object'):
        class_name = obj.find('name').text
        class_index = classes.index(class_name)
        bbox = obj.find('bndbox')
        xmin = float(bbox.find('xmin').text)
        ymin = float(bbox.find('ymin').text)
        xmax = float(bbox.find('xmax').text)
        ymax = float(bbox.find('ymax').text)

        # 计算中心坐标和宽度/高度的相对值
        x_center = (xmin + xmax) / (2.0 * image_width)
        y_center = (ymin + ymax) / (2.0 * image_height)
        width = (xmax - xmin) / image_width
        height = (ymax - ymin) / image_height

        # 将信息格式化为 YOLO 格式的一行文本
        yolo_line = f"{class_index} {x_center} {y_center} {width} {height}"
        yolo_lines.append(yolo_line)

    # 将 YOLO 格式的信息写入文本文件
    yolo_file_path = os.path.join(
        output_folder, os.path.splitext(image_filename)[0] + '.txt')
    with open(yolo_file_path, 'w') as yolo_file:
        for line in yolo_lines:
            yolo_file.write(line + '\n')


def convert_folder_to_yolo(xml_folder, classes, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    for xml_file in os.listdir(xml_folder):
        if xml_file.endswith('.xml'):
            xml_path = os.path.join(xml_folder, xml_file)
            convert_voc_to_yolo(xml_path, classes, output_folder)


# Example usage:
xml_folder_path = 'D:/Allpythonproject/cv-haut/Blood/archive (1)/dataset-master/dataset-master/Annotations'
output_folder_path = 'D:/Allpythonproject/cv-haut/Blood/archive (1)/dataset-master/dataset-master/output'
class_names = ["RBC"]

convert_folder_to_yolo(xml_folder_path, class_names, output_folder_path)
