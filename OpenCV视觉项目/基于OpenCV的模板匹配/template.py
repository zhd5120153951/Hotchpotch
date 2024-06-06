from pickletools import uint1
import cv2
import numpy as np
import time
import multiprocessing as mp
import base64
import requests

points = []


def mouse_callback(event, x, y, flag, param):
    global points
    if event == cv2.EVENT_LBUTTONDBLCLK:
        print(points)

    elif event == cv2.EVENT_LBUTTONDOWN:
        points.append([x, y])
        # Polygon_point.append(points)
    elif event == cv2.EVENT_RBUTTONDOWN:
        points.clear()
        # Polygon_point = []


def image_put(q, user, pwd, ip, channel=101):
    cap = cv2.VideoCapture(
        "rtsp://%s:%s@%s/Streaming/Channels/%d" % (user, pwd, ip, channel), cv2.CAP_FFMPEG)
    if cap.isOpened():
        print('HIKVISION')
    else:
        cap = cv2.VideoCapture(
            "rtsp://%s:%s@%s/cam/realmonitor?channel=%d&subtype=0" % (user, pwd, ip, channel))
        print('DaHua')

    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        q.put(frame)
        q.get() if q.qsize() > 4 else time.sleep(1)  # 保证帧队列中只有4帧


def image_get(q, window_name):
    global points
    cv2.namedWindow(window_name, flags=cv2.WINDOW_FREERATIO)
    cv2.resizeWindow(window_name, 1080, 1920)
    while True:
        frame = q.get()
        temp_frame = frame.copy()
        if len(points) > 0:
            cv2.polylines(temp_frame, np.array(
                [points]), isClosed=True, color=(0, 255, 0), thickness=2)
        cv2.imshow(window_name, temp_frame)
        cv2.setMouseCallback(window_name, mouse_callback)
        if cv2.waitKey(1) & 0xff == ord('q'):
            cv2.destroyAllWindows()
            break

    template_img = cv2.imread("./template.jpg")
    template_w, template_h, c = template_img.shape
    # 模版做掩码
    mask = np.zeros(
        [template_img.shape[0], template_img.shape[1]], dtype=np.uint8)
    mask = cv2.fillPoly(mask, [np.array(points)], (255, 255, 255))
    template_img = cv2.add(template_img, np.zeros(
        np.shape(template_img), dtype=np.uint8), mask=mask)
    cv2.imwrite('./template_mask.jpg', template_img)
    # 采用基于形状的模板匹配
    # 提取模板特征
    template_img_gray = cv2.cvtColor(template_img, cv2.COLOR_BGRA2GRAY)
    contours, hierarchy = cv2.findContours(
        template_img_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = contours[0] if len(contours) > 0 else None
    if contour is not None:
        pts = np.float32([[contour[0, 0], contour[0, -1]],
                         [contour[-1, 0], contour[0, -1]]])
        shape = cv2.convexHull(pts)
        shape = cv2.boundingRect(shape)
    else:
        shape = (0, 0, 0)
    cnt = 1
    while True:
        # 把输入做掩码
        frame = q.get()
        frame = cv2.add(frame, np.zeros(
            np.shape(frame), dtype=np.uint8), mask=mask)
        # 通用模板匹配
        # results = cv2.matchTemplate(frame, template_img, cv2.TM_CCOEFF_NORMED)
        # for y in range(len(results)):
        #     for x in range(len(results[y])):
        #         if results[y][x] > 0.99:  # 匹配成功--无堵塞物
        #             continue
        #         else:
        #             print("通道阻塞...")
        #             cv2.imwrite(f"./{cnt}.jpg", frame)
        #             cnt += 1

        # 基于形状特征的匹配
        results = cv2.matchTemplate(frame, shape, cv2.TM_CCOEFF_NORMED)
        loc = np.where(results >= 0.99)
        if len(loc) == 0:  # 不匹配
            cv2.imwrite(f'./{cnt}.jpg', frame)
            continue
        else:
            for pt in zip(*loc[::-1]):
                cv2.rectangle(
                    frame, pt, (pt[0]+shape[2], pt[1]+shape[3]), (0, 0, 255), 1)
                cv2.imwrite(f'./{cnt}.jpg', frame)
            cnt += 1

        # time.sleep(1)

        # cv2.imshow(window_name, frame)
        # cv2.waitKey(1)


def run_multi_camera(user_name, user_pwd):
    camera_ip_l = [
        "192.168.23.10",  # ipv4
        # "192.168.23.13",  # ipv4
        # "192.168.23.17",  # ipv4
        # "192.168.23.13",  # ipv4
        # "[fe80::3aaf:29ff:fed3:d260]",  # ipv6
    ]
    mp.set_start_method(method='spawn')  # init
    queues = [mp.Queue(maxsize=4) for _ in camera_ip_l]

    processes = []
    for queue, camera_ip in zip(queues, camera_ip_l):
        processes.append(mp.Process(target=image_put, args=(
            queue, user_name, user_pwd, camera_ip)))
        processes.append(mp.Process(target=image_get,
                         args=(queue, camera_ip)))

    for process in processes:
        process.daemon = True  # 主进程一死，这些子进程也被杀死
        process.start()
    for process in processes:
        process.join()


if __name__ == "__main__":
    run_multi_camera("admin", "jiankong123")
