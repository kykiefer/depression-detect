import os
import glob
import csv
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
            os.remove('static/matrices/{}'.format(npz_filename))  # delete npz

            return render_template('survey.html', spectrogram=spec_path, completion_status='Your audio has been successfully uploaded. Check out the visual representation below!')

    return render_template('donate.html')


@app.route('/thankyou', methods=['POST'])
def complete_survey():
    if request.method == 'POST':  # and survey filled out
        form = request.form
        if len(form.keys()) == 8:  # if form complete
            phq8_score = sum((int(j) for j in form.values()))

            # get the newest spectrogram upload to associate with depression label
            list_of_files = glob.glob('static/spectrograms/*.png')
            newest_partic = max(list_of_files, key=os.path.getctime)
            partic_id = os.path.split(newest_partic)[1]  # get file filename

            # append spectrogram identifier and phq8_score to csv
            fields = [partic_id, phq8_score]
            with open('dep_log.csv', 'a') as f:
                writer = csv.writer(f)
                writer.writerow(fields)

            # write csv to S3
            upload_file_to_s3('dep_log.csv')

            return render_template('thankyou.html')
        else:
            return render_template('survey.html', spectrogram='static/img/oops.png', completion_status='You did not fill in all the responses. Please complete the survey.')


@app.route('/contact')
def results():
    return render_template('contact.html')


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True, threaded=True)
