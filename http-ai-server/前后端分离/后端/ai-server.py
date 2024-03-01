from flask import Flask, request, jsonify
from flask_cors import CORS
app = Flask(__name__)
CORS(app)  # 处理跨域问题


@app.route('/proc_data', methods=['POST'])
def proc_data():
    print("接收到前端的请求了...")
    data_from_frontend = request.json['data']
    ret = data_from_frontend.upper()
    return jsonify({'ret': ret})


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5001, debug=True)
