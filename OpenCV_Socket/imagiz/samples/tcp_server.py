import imagiz
import cv2


def ServerRecieve():
    server = imagiz.TCP_Server(9990)
    server.start()
    while True:
        message = server.receive()
        # print(m)
        frame = cv2.imdecode(message.image, 1)
        cv2.imshow("", frame)
        cv2.waitKey(1)


if __name__ == "__main__":
    ServerRecieve()
