# Imagiz
Fast and none blocking live video streaming over network with OpenCV and (ZMQ or TCP).


# Install
```
pip3 install imagiz
```

# ZMQ Client

```
import imagiz
import cv2


client=imagiz.Client("cc1",server_ip="localhost")
vid=cv2.VideoCapture(0)
encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]

while True:
    r,frame=vid.read()
    if r:
        r, image = cv2.imencode('.jpg', frame, encode_param)
        client.send(image)
    else:
        break

```

# ZMQ Server
```
import imagiz
import cv2

server=imagiz.Server()
while True:
    message=server.recive()
    frame=cv2.imdecode(message.image,1)
    cv2.imshow("",frame)
    cv2.waitKey(1)
```
# TCP Client
```
import imagiz
import cv2

vid=cv2.VideoCapture(0)
client=imagiz.TCP_Client(server_port=9990,client_name="cc1")
encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]


while True:
    r,frame=vid.read()
    if r:
        r,image=cv2.imencode('.jpg',frame, encode_param)
        response=client.send(image)
        print(response)

```

# TCP Server
```
import imagiz
import cv2

server=imagiz.TCP_Server(9990)
server.start()
while True:
    message=server.receive()
    frame=cv2.imdecode(mmessage.image,1)
    cv2.imshow("",frame)
    cv2.waitKey(1)
```



# Client Options
| Name | Description |
| --- | --- |
| `client_name` | Name of client |
| `server_ip` | Ip of server default value is localhost  |
| `server_port` | Port of server default value is 5555 |
| `request_timeout` | how many milliseconds wait to resend image again |
| `request_retries` | how many time retries to send an image before client exit  |
| `generate_image_id` | image_id is unique and ordered id that you can use for save data on disk or db also it is show time of image sended by client  |
| `time_between_retries` | On TCP client show time between retries  |

# Server Options
| Name | Description |
| --- | --- |
| `Port` | Port of server |
| `listener` | Number of listening threads.default value is 10 |

# Message Class
| Name | Description |
| --- | --- |
| `image` | Byte of image |
| `client_name` | Name of client |
| `image_id` | If disable generate_image_id it will be 0  |
