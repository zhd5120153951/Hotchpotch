'''
@FileName   :modify.py
@Description:裁剪图像大小
@Date       :2023/02/25 12:07:49
@Author     :daito
@Website    :Https://github.com/zhd5120153951
@Copyright  :daito
@License    :None
@version    :1.0
@Email      :2462491568@qq.com
'''
import os
import cv2


def resize_img(DATADIR, data_k, img_size):
    w = img_size[0]
    h = img_size[1]
    path = os.path.join(DATADIR, data_k)
    #返回path路劲下的全部文件名字或者文件夹的名字
    img_list = os.listdir(path)

    for i in img_list:
        if i.endswith('.jpg'):
            #发现jpg-读取
            img_array = cv2.imread((path + '/' + i), cv2.IMREAD_COLOR)
            #resize()
            new_array = cv2.resize(img_array, (w, h), interpolation=cv2.INTER_CUBIC)
            img_name = str(i)
            #生成图片的存储路劲
            save_path = path + '_new/'
            if os.path.exists(save_path):
                print(i)
                #调用写图片
                save_img = save_path + img_name
                cv2.imwrite(save_img, new_array)
            else:
                os.mkdir(save_path)
                save_img = save_path + img_name
                cv2.imwrite(save_img, new_array)


if __name__ == '__main__':
    #路径

    #DATADIR = "E:/test/LeNet-5/data/LEDNUM/train_data/"
    DATADIR = "E:/test/LeNet-5/data/LEDNUM/test_data/"
    img_size = [320, 320]
    for i in range(0, 10):
        data_k = str(i)
        resize_img(DATADIR, data_k, img_size)
