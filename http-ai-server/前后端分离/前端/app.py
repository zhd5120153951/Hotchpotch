from flask import Flask, render_template

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/camera')
def camera():
    return render_template('camera.html')


@app.route('/preview')
def preview():
    return render_template('preview.html')


if __name__ == '__main__':
    app.run(debug=True)
