from flask import Flask, Response, render_template, request, jsonify, url_for
from flask_cors import CORS
import cv2
import requests
import sqlite3


app = Flask(__name__)
CORS(app)


def connSqlite(sqlitePath):
    conn = sqlite3.connect(sqlitePath)
    cursor = conn.cursor()
    return conn, cursor


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/camera')
def camera():
    return render_template('camera.html')


@app.route('/addcamera', methods=['POST'])
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


@app.route('/preview')
def preview():
    return render_template('preview.html')
    # return render_template('prev.html')


@app.route('/preview/getcamera', methods=['GET'])
def getCamera():
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


@app.route('/preview/modifycamera', methods=['POST'])
def modifyCamera():
    print("更新相机配置")
    return jsonify({'success': True, 'msg': '模拟更改成功'})


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


@app.route('/control')
def control():
    return render_template('control.html')


@app.route('/warn')
def warn():
    return render_template('warn.html')


@app.route('/editcontrol')
def config_control():
    return render_template('editcontrol.html')


class VideoCamera(object):
    def __init__(self, rtspurl):
        # 通过opencv获取实时视频流
        self.video = cv2.VideoCapture(rtspurl)

    def __del__(self):
        self.video.release()

    def get_frame(self):
        success, image = self.video.read()
        # 因为opencv读取的图片并非jpeg格式，因此要用motion JPEG模式需要先将图片转码成jpg格式图片
        jpeg = cv2.imencode('.jpg', image)[1].tobytes()
        return jpeg


def video_gen(camera):
    cap = cv2.VideoCapture(camera)
    while cap.isOpened():
        print("while>>>>>>>>>>>>>>>>>>>>>>")
        ret, frame = cap.read()
        # 使用generator函数输出视频流， 每次请求输出的content类型是image/jpeg
        if not ret:
            continue
        frame = cv2.imencode('.jpg', frame)[1].tobytes()
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/video_view')
def video_view():
    return Response(video_gen(0), mimetype='multipart/x-mixed-replace;bounddary=frame')


if __name__ == '__main__':
    app.run(debug=True)
