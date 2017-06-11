import boto
from boto.s3.key import Key
import os
from flask import Flask, render_template, flash, request, redirect, url_for
from werkzeug.utils import secure_filename
from spectrogram import plotstft
access_key = os.environ['AWS_ACCESS_KEY_ID']
access_secret_key = os.environ['AWS_SECRET_ACCESS_KEY']

UPLOAD_FOLDER = 'static/audio_uploads'
ALLOWED_EXTENSIONS = 'wav'

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/donate', methods=['GET', 'POST'])
def upload_file():
    #  if user has submitted the form
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            # saving to static/audio_uploads temporarily
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

            # save spectrogram to static/spectrograms
            plotstft(os.path.join(app.config['UPLOAD_FOLDER'], filename),
                     plotpath='static/spectrograms/{}.png'.format(filename))

            # get matrix reprentation and store to s3
            # delete wav file

            spec = 'static/spectrograms/{}.png'.format(filename)

            # # connect to S3
            # conn = boto.connect_s3(access_key, access_secret_key)
            #
            # # get handle to the S3 bucket
            # bucket_name = 'depression-detect'
            # bucket = conn.get_bucket(bucket_name)

            # file_object = bucket.new_key(filename)
            # file_object.set_contents_from_filename(filename)
            return render_template('survey.html', spectrogram=spec)

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


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True, threaded=True)
