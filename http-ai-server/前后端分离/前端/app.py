from flask import Flask, render_template, Response
import cv2
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
    # return render_template('prev.html')


@app.route('/control')
def control():
    return render_template('control.html')


@app.route('/warn')
def warn():
    return render_template('warn.html')


@app.route('/editcontrol')
def config_control():
    return render_template('editcontrol.html')


# def video_gen(rtspurl):
#     cap = cv2.VideoCapture(rtspurl)
#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             cap.release()
#             cap = cv2.VideoCapture(rtspurl)
#         frame = cv2.imencode('.jpg', frame)[1].tobytes()
#         yield (b'--frame\r\n' b'Content-Type:image/jpeg\r\n\r\n'+frame+b'\r\n')


# @app.route('/video_view/<feed_type>')
# def video_view(feed_type):
#     return Response(video_gen(feed_type), mimetype='multipart/x-mixed-replace; bounddary=frame')


if __name__ == '__main__':
    app.run(debug=True)
