from flask import Flask, render_template, request, make_response
from werkzeug.utils import secure_filename
from os.path import abspath
from model import resnet50_predict

app = Flask(__name__, static_url_path="", static_folder="static")


@app.route('/')
def index():
    return render_template("index.html")

@app.route("/project")
def project():
    return render_template("project.html", result={})

@app.route('/uploader', methods = ['POST'])
def upload_file():
   if request.method == 'POST':
      f = request.files['file']
      path = abspath("uploads/" + secure_filename(f.filename))
      f.save(path)


      # path 为上传图片的路径，然后将对于的路径发送到对于的函数。
      # path = somefunction(path) 返回的值为对于处理的函数后的图片的路径。
      # 此时path为对于的函数处理后的程序。

      # resp = make_response(open(path, mode="rb").read())
      # resp.content_type = "image/jpeg"
      # return resp

      # 机器学习处理代码
      # result = process(path)
      path = resnet50_predict.predict_API(path)
      result = {"path": path}
      return render_template("project.html", result)

if __name__ == '__main__':
    app.run("0.0.0.0")
