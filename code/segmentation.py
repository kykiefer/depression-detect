from pyAudioAnalysis import audioBasicIO as aIO
from pyAudioAnalysis import audioSegmentation as aS
import scipy.io.wavfile as wavfile
import os


def silence_extraction(filename, smoothing=1.0, weight=0.3, plot=False):
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
    [Fs, x] = aIO.readAudioFile(filename)
    segments = aS.silenceRemoval(x, Fs, 0.020, 0.020, smoothWindow=smoothing, Weight=weight, plot=plot)

    for s in segments:
        seg_name = "{0:s}_{1:.2f}-{2:.2f}.wav".format('ky', s[0], s[1])
        print(seg_name)
        # wavfile.write(seg_name, Fs, x[int(Fs * s[0]):int(Fs * s[1])])

if __name__ == '__main__':
    dir_name = '/Users/ky/Desktop/depression-detect/raw_data/audio_test' # directory containing wav files
    extension = ".wav"

    for file in os.listdir(dir_name):
        if file.endswith(extension):
            filename = os.path.join(dir_name, file)
            print(filename)
            silence_extraction(filename)
