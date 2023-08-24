import numpy as np
import cv2
import time
import pandas as pd
import joblib
import argparse


def load_data(feature_file, mark):
    feature = pd.read_csv(feature_file, sep=",", header=None)
    feature = np.array(feature)
    if mark == 0:  #nofire--背景
        return np.concatenate((feature, np.zeros((feature.shape[0], 1))), axis=1)
    if mark == 1:  #fire
        return np.concatenate((feature, np.ones((feature.shape[0], 1))), axis=1)


def model_train(model_name="GaussianNB"):
    feature0 = load_data("images/data/fire.txt", 1)  #fire
    feature1 = load_data("images/data/nofire.txt", 0)  #nofire--背景
    feature = np.array(np.concatenate([feature0, feature1], axis=0), dtype=np.uint8)
    #feature = pd.DataFrame(np.array(np.concatenate([feature0, feature1], axis = 0), dtype=np.uint8))
    #feature[0], feature[3] = feature[3], feature[0]
    #feature.to_csv("out.csv", index=False)
    feature = feature.astype(np.float32)
    data = feature[:, :-1]
    label = feature[:, -1:].flatten()

    if model_name == "SVC":
        from sklearn.svm import SVC
        model = SVC()

    model.fit(data, label)
    model_name = f"{model}"[:-2]
    joblib.dump(model, f'./images/models/{model_name}.pkl')

    test0 = load_data("./images/data/testfire.txt", 1)
    datatest0 = test0[:, :-1]
    labeltest0 = test0[:, -1:].flatten()
    result = model.predict(datatest0)
    count = (labeltest0 == result).sum()
    correct_rate = count / len(datatest0)
    print("label 0 test correct_rate ", correct_rate)

    test1 = load_data("./images/data/testnofire.txt", 0)
    datatest1 = test1[:, :-1]
    labeltest1 = test1[:, -1:].flatten()
    result = model.predict(datatest1)
    count = (labeltest1 == result).sum()
    correct_rate = count / len(datatest1)
    print("label 1 test correct_rate ", correct_rate)


def model_predict(model_name="GaussianNB", pic_name='images/test/3.jpg'):

    model2 = joblib.load(f'./images/models/{model_name}.pkl')
    test_img = cv2.imread(pic_name)

    size = list(test_img.shape)
    size[2] = 1

    window_name2 = 'pretmp'
    cv2.namedWindow(window_name2, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name2, 540, 540)
    window_name = 'pre'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 540, 540)
    window_name1 = 'src'
    cv2.namedWindow(window_name1, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name1, 540, 540)

    mark_time1 = time.time()
    pre = model2.predict(np.array(test_img).reshape(-1, 3))
    pre = pre.reshape([size[0], size[1]])
    pretmp = pre
    cv2.imshow("pretmp", pretmp)

    kernel = np.ones((3, 3), dtype=np.uint8)
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    pre = cv2.dilate(cv2.erode(pre, kernel), kernel)

    mark_time2 = time.time()
    print(f"{model_name}模型推理用时: {mark_time2 - mark_time1}")

    cv2.imshow("src", test_img)
    cv2.imshow("pre", pre)
    pre = pre * 255

    #连通操作
    pre = pre.astype(np.uint8)  #需要把类型转换才行
    print(pre.shape)
    contours, _ = cv2.findContours(pre, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        area = cv2.contourArea(contour)
        print(area)
        if area >= 600:
            print('fire')
            cv2.drawContours(test_img, contour, -1, (0, 255, 0), 1)
            cv2.imshow('draw', test_img)

    pre = np.expand_dims(pre, axis=2)
    print(pre.shape)
    tmppic = np.concatenate([pre, pre, pre], axis=2)
    cv2.imwrite(f"./images/tmp/{model_name}_res.jpg", tmppic)

    pretmp = pretmp * 255
    pretmp = np.expand_dims(pretmp, axis=2)
    pixelclassify = np.concatenate([pretmp, pretmp, pretmp], axis=2)
    cv2.imwrite(f"./images/tmp/{model_name}_restmp.jpg", pixelclassify)

    cv2.waitKey(0)


def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', nargs='+', default=['SVC'], help='SVC')
    args = parser.parse_args()
    print(args)
    return args


if __name__ == "__main__":
    args = parse_opt()

    model_name = args.model_name[0]

    #job 1 train the pixels
    # model_train(model_name)
    #job 2 test the job2's model
    model_predict(model_name)
