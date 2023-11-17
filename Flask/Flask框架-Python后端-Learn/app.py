'''
@FileName   :app.py
@Description:
@Date       :2023/11/16 17:00:52
@Author     :daito
@Website    :Https://github.com/zhd5120153951
@Copyright  :daito
@License    :None
@version    :1.0
@Email      :2462491568@qq.com
'''

from flask import Flask, request

app = Flask(__name__)

# 路由函数(url地址)


@app.route('/')
def hello_world():
    return "hello world"


@app.route('/profile')
def profile():
    return "welcome to my profile"

# 带参数--是指url地址


@app.route('/list/<int:num>')
def list_num(num):
    return "welcome to my profile:%s" % num


# 带参数--是指book/list?page=123


@app.route('/book/list')
def book_list():
    page = request.args.get("page", default=1, type=int)

    return f"你获取的是的第{page}页的图书列表..."


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
