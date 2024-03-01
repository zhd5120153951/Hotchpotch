from flask import Flask, render_template, request, jsonify

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('home.html')

# 前后端不分离--前端和后端写在一个py中,请求地址也不同


@app.route('/process_data', methods=['POST'])
def process_data():
    # 从前端获取参数
    data_from_frontend = request.form['data']

    # 在这里可以对接收到的参数进行处理，比如执行某个算法
    # 示例中简单地将参数转为大写
    result = data_from_frontend.upper()

    # 返回处理后的结果给前端
    return jsonify({'result': result})


if __name__ == '__main__':
    app.run(debug=True)
