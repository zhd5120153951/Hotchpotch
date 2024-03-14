import json
from turtle import rt
import cv2
from flask import Flask, request, jsonify, Response
from flask_cors import CORS
import requests
import sqlite3

app = Flask(__name__)
CORS(app)  # 处理跨域问题


@app.route('/proc_data', methods=['POST'])
def proc_data():
    print("接收到前端的请求了...")
    data_from_frontend = request.json['data']
    ret = data_from_frontend.upper()
    print("发送给AI端")

    res = requests.post("http://127.0.0.1:9003/image/objectDetect", data={
        "appKey": "appKey",
        "image_base64": "images",
        "algorithm": "openvino",
    })
    if 200 == res.status_code:  # AI端正常响应
        data = res.json()
        code = data.get("code")
        msg = data.get("msg")
        print(f"code:{code}\tmsg:{msg}")

    return jsonify({'ret': ret})


@app.route('/config', methods=['POST'])
def send_config():
    print("接收到config")
    rtsp = request.json['rtspurl']
    thresh = request.json['thresh']
    detInterval = request.json['detInterval']
    print(f"rtsp:{rtsp}\tthresh:{thresh}\tdetInterval:{detInterval}")
    ret = requests.post("http://127.0.0.1:9003/image/objectDetect", data={
        "appKey": "appKey",
        "image_base64": "image",
        "algorithm": "openvino",
    })
    if 200 == ret.status_code:
        data = ret.json()
        code = data.get('code')
        msg = data.get('msg')
        print(f"code:{code}\tmsg{msg}")
    return jsonify({'success': True, 'msg': '已处理完算法'})


def connSqlite(sqlitePath):
    conn = sqlite3.connect(sqlitePath)
    cursor = conn.cursor()
    return conn, cursor


@app.route('/camera', methods=['POST'])
def addCamera():
    # 操作把数据写入数据库
    # 1.先查询是否存在同样的摄像头
    usernmae = request.json['username']
    password = request.json['password']
    rtspurl = request.json['rtspurl']
    conn, cursor = connSqlite('admin.db')
    cursor.execute("SELECT * FROM camera")
    rowAll = cursor.fetchall()

    for i in range(len(rowAll)):
        if rtspurl == rowAll[i][2]:
            return jsonify({'success': False, 'msg': '该摄像头已经存在,请重新设置.'})
        else:
            continue
    cursor.execute('''INSERT INTO camera (username,password,rtspurl) VALUES (?,?,?)''',
                   (usernmae, password, rtspurl))

    # 提交
    conn.commit()
    conn.close()
    return jsonify({'success': True, 'msg': '该摄像头配置成功'})


@app.route('/preview', methods=['GET'])
def prevCamera():
    data = []
    # 查询数据库中的rtsp
    conn, cursor = connSqlite('admin.db')
    cursor.execute("SELECT * FROM camera")
    rowAll = cursor.fetchall()
    for i in range(len(rowAll)):
        data.append(
            {f"id": i+1, "username": rowAll[i][0], "password": rowAll[i][1], "rtspurl": rowAll[i][2]})
    conn.close()
    # print(data)
    return data
    # return jsonify({"username": "daito", "password": "daito", "email": "2462491568@qq.com"})


@app.route('/preview/deleteURL', methods=['POST'])
def deleteURL():
    rtspurl = request.json['rtspurl']
    print(rtspurl)
    if rtspurl:
        # 查询数据库中的rtsp
        conn, cursor = connSqlite('admin.db')

        cursor.execute("DELETE FROM camera WHERE rtspurl=?",
                       (rtspurl,))  # 只有一个条件时,必须这个写
        conn.commit()
        conn.close()
        return jsonify({'success': True, 'msg': '成功删除该摄像头'})
    return jsonify({'success': False, 'msg': '流地址为空,请选择一个流地址'})


@app.route('/video_frame', methods=['POST'])
def video_frame():
    global rtspurl
    rtspurl = request.json['rtspurl']
    print(rtspurl)
    if rtspurl:
        return jsonify({'success': True, 'msg': rtspurl})
    return jsonify({'success': False, 'msg': '流地址出错'})


class VideoCamera(object):
    def __init__(self, rtspurl):
        # 通过opencv获取实时视频流
        self.video = cv2.VideoCapture(rtspurl)

    def __del__(self):
        self.video.release()

    def get_frame(self):
        success, image = self.video.read()
        # 因为opencv读取的图片并非jpeg格式，因此要用motion JPEG模式需要先将图片转码成jpg格式图片
        ret, jpeg = cv2.imencode('.jpg', image)
        return jpeg.tobytes()


def video_gen(camera):
    while True:
        frame = camera.get_frame()
        # 使用generator函数输出视频流， 每次请求输出的content类型是image/jpeg
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


@app.route('/video_view')
def video_view():
    global rtspurl
    print(rtspurl, 2)
    return Response(video_gen(VideoCamera(rtspurl)), mimetype='multipart/x-mixed-replace;bounddary=frame')


if __name__ == "__main__":
    rtspurl = None
    app.run(host='0.0.0.0', port=5001, debug=True)
