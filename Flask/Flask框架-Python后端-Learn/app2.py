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

from flask import Flask, request, render_template

app = Flask(__name__)

# 后端向前端html传递一个对象


class User():
    def __init__(self, username, email) -> None:
        self.username = username
        self.email = email


# 路由函数(url地址)


@app.route("/")
def index():
    # 得到对象
    user = User(username="daito", email="XX@greatech.com")

    person = {
        "username": "lucy",
        "email": "adad@qq.com"
    }
    return render_template("app2/index.html", user=user, person=person)


@app.route("/blog/<blog_id>")
def blog_list(blog_id):
    return render_template("app2/blog_list.html", blog_id=blog_id)


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
