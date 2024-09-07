import imagiz
import cv2

# 客户端采集往服务端发


def ClientSend(rtsp):
    vid = cv2.VideoCapture(rtsp)
    client = imagiz.TCP_Client(server_port=9990, client_name="cc1")
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]

    while True:
        r, frame = vid.read()
        if r:
            r, image = cv2.imencode('.jpg', frame, encode_param)
            response = client.send(image)
            print(response)


if __name__ == "__main__":
    rtsp = "rtsp://admin:jiankong123@192.168.23.15:554/Streaming/Channels/101"
    ClientSend(rtsp)
