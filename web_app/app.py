import os
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from spectrogram import plotstft
import numpy as np
from upload_to_s3 import upload_file_to_s3
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
    #  if user has submitted the audio file
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            wav_filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

            # save wav file  to static/audio_uploads temporarily
            file.save(wav_filepath)

            # save spectrogram to static/spectrograms
            png_filename = os.path.splitext(filename)[0]+'.png'
            spec_path = 'static/spectrograms/{}'.format(png_filename)

            # plot spectrogram and return matrix
            spec_matrix = plotstft(wav_filepath, plotpath=spec_path)

            # save matrix locally
            npz_filename = os.path.splitext(filename)[0]+'.npz'
            np.savez('static/matrices/{}'.format(npz_filename), spec_matrix)

            # upload matrix to s3
            upload_file_to_s3('static/matrices/{}'.format(npz_filename))

            os.remove(wav_filepath)  # delete wav file
            os.remove('static/matrices/{}'.format(npz_filename))  # delete local npz file

            return render_template('survey.html', spectrogram=spec_path)

    return render_template('donate.html')


@app.route('/survey', methods=['POST'])
def survey():
        return render_template('thank_you.html')


@app.route('/thank_you', methods=['GET', 'POST'])
def thank_you():
        return render_template('thank_you.html')


@app.route('/contact')
def results():
    return render_template('contact.html')


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True, threaded=True)
