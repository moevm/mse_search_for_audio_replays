from flask import Flask, render_template, url_for, request, make_response
import os

UPLOAD_FOLDER = 'tmp/'

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route('/')
def index():
    return render_template("index.html")


@app.route('/upload', methods=["POST"])
def upload():
    if request.method == "POST":
        files = request.files.getlist('audio_files')
        print(files)
        for file in files:
            if file.filename != '':
                #file = files.read()
                #tmp = open("/tmp/" + files.filename, "w")
                #print(file, file=tmp)
                #tmp.close()
                file.save(os.path.join(app.config['UPLOAD_FOLDER'], file.filename))
                print(os.path.join(app.config['UPLOAD_FOLDER'], file.filename))
        return make_response("OK!", 200)
    else:
        return "Ошибка запроса"


if __name__ == "__main__":
    app.run(debug=True)
    print(app)
