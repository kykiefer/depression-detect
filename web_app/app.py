import boto
from boto.s3.key import Key
import os
from flask import Flask, render_template, flash, request, redirect, url_for
from werkzeug.utils import secure_filename
access_key = os.environ['AWS_ACCESS_KEY_ID']
access_secret_key = os.environ['AWS_SECRET_ACCESS_KEY']

UPLOAD_FOLDER = 'Users/ky/Desktop/'
ALLOWED_EXTENSIONS = set(['wav', 'jpg', 'png'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/donate')
def donate():
    return render_template('donate.html')


@app.route('/survey', methods=['POST'])
def survey():
    return render_template('survey.html')


@app.route('/contact')
def results():
    return render_template('contact.html')


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# Route that will process the file upload
@app.route('/upload', methods=['POST'])
def upload():
    string = ''
    #  if user has submitted the form
    if request.method == 'POST':
        string += 'post'
        submitted_file = request.files['file']
        if submitted_file and allowed_file(submitted_file.filename):
            filename = secure_filename(submitted_file.filename)
            return filename
            # connect to S3
            conn = boto.connect_s3(access_key, access_secret_key)

            # get handle to the S3 bucket
            bucket_name = 'depression-detect'
            bucket = conn.get_bucket(bucket_name)

            file_object = bucket.new_key(filename)
            file_object.set_contents_from_filename(filename)

    return string


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True, threaded=True)
