from flask import Flask, render_template, url_for, request, redirect, session, flash, Response
from flask_sqlalchemy import SQLAlchemy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils import predictsenti, predictActigraph, FetureExtraction, sentipredictbulk, actibulkpredic
import os
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import io
import time
from werkzeug.utils import secure_filename
import glob


ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif', 'csv'}
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "Uploads")

if not os.path.exists(UPLOAD_FOLDER):
    os.mkdir(UPLOAD_FOLDER)


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + \
    BASE_DIR + os.path.sep + "test.db"
db = SQLAlchemy(app)
app.secret_key = "abc"


def newest(path):
    files = os.listdir(path)
    paths = [os.path.join(path, basename) for basename in files]
    return max(paths, key=os.path.getctime)


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def create_figure():
    fig = Figure()
    last_saved = db.engine.execute(
        "select max(id + 1 ) as id, file_name from Temp_storage")
    for i in last_saved:
        print(i[0])
        print(i[1])
        last_saved_file_name = i[1]
    csv_path = os.path.join(UPLOAD_FOLDER, last_saved_file_name)
    print("csv_path: ", csv_path)
    axis = fig.add_subplot(1, 1, 1)
    print("CSV Path: ", csv_path)
    df = pd.read_csv(csv_path, parse_dates=[
                     "timestamp"], index_col="timestamp")
    df = df.head(60)
    print(df.head())
    axis.plot(df['activity'].values)
    return fig


class user(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    fullname = db.Column(db.String(120))
    username = db.Column(db.String(80))
    email = db.Column(db.String(120))
    password = db.Column(db.String(80))


class Temp_storage(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    file_name = db.Column(db.String(150), nullable=False)


@app.route('/plot.png')
def plot_png():
    fig = create_figure()
    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)
    return Response(output.getvalue(), mimetype='image/png')


@app.route('/ajaxImage')
def bring_image():
    return render_template("ajax_image.html")


@app.route('/', methods=["GET", "POST"])
def index():
    error = ""
    if request.method == "POST":
        uname = request.form["username"]
        passw = request.form["pass"]
        login = user.query.filter_by(username=uname, password=passw).first()
        if login is not None:
            session['logged_in'] = True
            session['username'] = login.username
            user.authenticated = True
            print("Login Succesfully")
            return render_template('home.html')
        else:
            error = "Invalid username or password"
    return render_template('index.html', error=error)


@app.route('/home')
def home():
    return render_template('home.html')


@app.route('/newlog')
def regpage():
    return render_template('newlog.html')


@app.route('/ragister', methods=['POST'])
def ragister():
    if request.method == "POST":
        print(request.form)
        first = request.form['firstname']
        last = request.form['lastname']
        fn = ' '.join([first, last])
        uname = request.form['username']
        mail = request.form['email']
        passw = request.form['pass']
        dup = user.query.filter_by(username=uname).first()
        if dup is not None:
            faild = "User Allready Registerd..!"
            print(faild)
            return render_template('newlog.html', faild=faild)
        else:
            register = user(fullname=fn, username=uname,
                            email=mail, password=passw)
            db.session.add(register)
            db.session.commit()
            success = "Profile Registration is Succesfull..!"
            print(success)
            return render_template('newlog.html', success=success)


@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/actigraph')
def actigraph():
    return render_template('actigraph.html')


@app.route('/actiPredict', methods=["POST", "GET"])
def actiPredict():
    new_filename = ""
    if request.method == "POST":
        file = request.files['file']
        print(request.files)
        if file.filename != "":
            if file and allowed_file(file.filename):
                file_name, file_ext = os.path.splitext(file.filename)
                new_filename = file_name + \
                    time.strftime("-%Y-%d-%M-%H-%S") + file_ext
                file.filename = new_filename
                filename = secure_filename(file.filename)
                file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
                add_this = Temp_storage(file_name=new_filename)
                db.session.add(add_this)
                db.session.commit()
                last_saved = db.engine.execute(
                    "select max(id + 1 ) as id, file_name from Temp_storage")
                for i in last_saved:
                    print(i[0])
                    print(i[1])
                    last_saved_file_name = i[1]
                csv_path = os.path.join(UPLOAD_FOLDER, last_saved_file_name)
                df = pd.read_csv(csv_path, parse_dates=[
                                 "timestamp"], index_col="timestamp")
                result = predictActigraph(df)
                print(result)
                return render_template("actigraph.html", data="got_image", result=result)

    return render_template("actigraph.html", data="no_image")


@app.route('/senti')
def senti():
    return render_template('senti.html')


@app.route('/sentibulk')
def sentibulk():
    return render_template('sentibulk.html')


@app.route('/sentiPredict', methods=['POST'])
def sentiPredict():
    if request.method == "POST":
        print(request.form)
        text = request.form['text']
        result = predictsenti(text)
        print(result)
    return render_template('senti.html', result=result, text=text)


@app.route('/sentibulkpredication', methods=['POST'])
def sentibulkpredication():
    if request.method == "POST":
        print(request.form)
        file = request.files['file']
        if file:
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        file_new = newest(UPLOAD_FOLDER)
        df = pd.read_csv(file_new)
        result=sentipredictbulk(df)
        print(result.head(5))
    return render_template('sentibulk.html', tables=[result.to_html(classes='data')], titles=result.columns)


@ app.route('/actibulk')
def actibulk():
    return render_template('actibulk.html')


@ app.route('/actibulkpredication', methods=['GET', 'POST'])
def actibulkpredication():
    if request.method == 'POST':
        files=request.files.getlist("file")
        for file in files:
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], file.filename))
        files_path=os.path.join(UPLOAD_FOLDER, '*')
        f=sorted(glob.iglob(files_path), key=os.path.getctime, reverse=True)
    print(f[0:len(files)])
    result=actibulkpredic(f[0:len(files)])
    return render_template('actibulk.html', tables=[result.to_html(classes='data')], titles=result.columns)


@ app.route('/logout')
def logout():
    # remove the username from the session if it is there
    session.pop('username', None)
    user.authenticated=False
    return redirect(url_for('index'))


@ app.route('/userinfo')
def userinfo():
    users=user.query.all()
    return render_template('userinfo.html', users=users, title="Show Users")


@ app.route('/userupdatepage')
def userupdatepage():
    users=user.query.all()
    return render_template('userinfo.html', users=users, title="Show Users")


@ app.route('/userupdate', methods=["POST"])
def userupdate():
    if request.method == "POST":
        print(request.form)
        username=request.form['username']
        fullname=request.form['fullname']
        email=request.form['email']
        passw=request.form['pass']
        update=user.query.filter_by(username=username).first()
        update.fullname=fullname
        update.email=email
        update.password=passw
        db.session.commit()
        success="Information Update Sccuesfully..!"
        print(success)
        users=user.query.all()
        return render_template('userinfo.html', users=users, success=success)


@ app.route('/help')
def help():
    return render_template('help.html')


@ app.errorhandler(404)
def not_found(e):
    return render_template("404.html")


if __name__ == '__main__':
    db.create_all()
    app.run(debug=True)
