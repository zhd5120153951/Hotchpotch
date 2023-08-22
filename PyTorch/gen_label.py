'''
@FileName   :gen_label.py
@Description:生成图像数据对应的标签文件
@Date       :2023/03/04 09:27:46
@Author     :daito
@Website    :Https://github.com/zhd5120153951
@Copyright  :daito
@License    :None
@version    :1.0
@Email      :2462491568@qq.com
'''
import os
'''
    为数据集生成对应的txt文件
'''

#path = 'F:/Code/Object Detection/data/LEDNUM/'
#src = os.path.join(os.path.abspath(path), item)

train_txt_path = os.path.join("DataSet", "LEDNUM", "train.txt")
train_dir = os.path.join("DataSet", "LEDNUM", "train_data")

valid_txt_path = os.path.join("DataSet", "LEDNUM", "test.txt")
valid_dir = os.path.join("DataSet", "LEDNUM", "test_data")


#生成标签
def gen_txt(txt_path, img_dir):
    f = open(txt_path, 'w')

    for root, s_dirs, _ in os.walk(img_dir, topdown=True):  # 获取img_dir文件下各文件夹名称
        for sub_dir in s_dirs:
            i_dir = os.path.join(root, sub_dir)  # 获取各类的子文件夹 绝对路径
            img_list = os.listdir(i_dir)  # 获取类别文件夹下所有jpg图片的路径
            for i in range(len(img_list)):
                if not img_list[i].endswith('jpg'):  # 若不是jpg文件，跳过
                    continue
                label = img_list[i].split('_')[0]  # 比如：0_34.jpg-->['0','34.jpg']
                img_path = os.path.join(i_dir, img_list[i])
                line = img_path + ' ' + label + '\n'
                f.write(line)
    f.close()


if __name__ == '__main__':
    gen_txt(train_txt_path, train_dir)
    gen_txt(valid_txt_path, valid_dir)