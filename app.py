from flask import Flask, render_template, url_for, request, make_response
from flask_socketio import emit, SocketIO
import os
import src.main

UPLOAD_FOLDER = 'tmp/'

app = Flask(__name__)
socketio = SocketIO(app)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route('/')
def index():
    return render_template("index.html")


@app.route('/upload', methods=["POST"])
def upload():
    if request.method == "POST":
        files = request.files.getlist('audio_files')
        flag = request.form['mode']
        print(flag)
        for file in files:
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], file.filename))
            print(os.path.join(app.config['UPLOAD_FOLDER'], file.filename))
        return make_response()
    else:
        return "Ошибка запроса"


@socketio.on("connect")
def connected():
    print("client connected")


@socketio.on("disconnect")
def disconnected():
    print("client disconnected")


if __name__ == "__main__":
    app.run(debug=True)
    print(app)
