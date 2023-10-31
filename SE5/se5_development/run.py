import argparse
import cv2
import json


def start_inference(config_file):
    with open(config_file) as file:
        data = json.load(file)
    rtsp_ip = data['rtsp_ip']
    save_img_path = data['save_img_path']

    cap = cv2.VideoCapture(rtsp_ip)
    frame_gap = 100
    cn = 1
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        #跳帧
        if frame_gap > 1:
            for _ in range(frame_gap - 1):
                ret, _ = cap.read()
                if not ret:
                    break
        cv2.imwrite(save_img_path + "/{}.jpg".format(cn), frame)
        cn += 1
        if cn > 150:
            break


def main():
    parser = argparse.ArgumentParser(description="YOLO Inference with Config")
    parser.add_argument("--config",
                        type=str,
                        default='/data/greatech/step_1/local.json',
                        help="Path to the config file")
    args = parser.parse_args()

    config_file = args.config
    start_inference(config_file)


if __name__ == "__main__":
    main()
