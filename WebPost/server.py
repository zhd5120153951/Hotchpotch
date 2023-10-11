#服务器端先运行server.py,发送端再运行client.py
from flask import request, Flask
import base64

app = Flask(__name__)


@app.route("/", methods=['POST', 'GET'])
def get_frame():
    # 解析图片数据
    # img = base64.b64decode(str(request.form['file1']))
    # img=str(request.form['file1'])
    file = request.files['file']
    file.save('test.jpg')
    return {'sim': "0.8"}


if __name__ == "__main__":
    app.run("0.0.0.0", port=6666)
