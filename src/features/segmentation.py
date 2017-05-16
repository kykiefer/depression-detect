from pyAudioAnalysis import audioBasicIO as aIO
from pyAudioAnalysis import audioSegmentation as aS
import scipy.io.wavfile as wavfile
import os


def silence_extraction(filename, out_dir, smoothing=1.0, weight=0.3, plot=False):
    """
    A function that implements pyAudioAnalysis' silence extraction and creates wav files of the non-silent portions of audio. The smoothing and weight parameters were tuned for the AVEC 2016 data set.

    Parameters
    ----------
    filename : string
        the input wav file
    smoothing : float
        ??????desired frame size in milliseconds
    weight : float
        ??????desired frame step in milliseconds
    """
    # create audio_segments directory and change to it
    audio_segments_dir = os.path.join(out_dir, 'audio_segments')
    if not os.path.exists(audio_segments_dir):
        os.makedirs(audio_segments_dir)

    os.chdir(audio_segments_dir)

    [Fs, x] = aIO.readAudioFile(filename)
    segments = aS.silenceRemoval(x, Fs, 0.020, 0.020, smoothWindow=smoothing, Weight=weight, plot=plot)

    partic_id = 'P' + filename.split('/')[-1].split('_')[0] # PXXX
    for s in segments:
        seg_name = "{:s}_{:.2f}-{:.2f}.wav".format(partic_id, s[0], s[1])
        wavfile.write(seg_name, Fs, x[int(Fs * s[0]):int(Fs * s[1])])


if __name__ == '__main__':
    dir_name = '/Users/ky/Desktop/depression-detect/raw_data/audio' # directory containing wav files
    extension = ".wav"
    out_dir = '/Users/ky/Desktop/depression-detect/raw_data'

    for file in os.listdir(dir_name):
        if file.endswith(extension):
            filename = os.path.join(dir_name, file)
            silence_extraction(filename, out_dir=out_dir)
