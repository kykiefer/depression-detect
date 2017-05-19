""" This work is licensed under a Creative Commons Attribution 3.0 Unported License. Frank Zalkow, 2012-2013 """

"""
This script creates spectrograms from wave files that can be passed to the CNN. This was heavily adapted form Frank Zalkow's work. I just dropped the matplotlib plotting and added in the PIL functionality so that the .pngs can be passed to a CNN.
"""

from matplotlib import pyplot as plt
import numpy as np
from numpy.lib import stride_tricks
import os
from PIL import Image
import scipy.io.wavfile as wav


""" short time fourier transform of audio signal """
def stft(sig, frameSize, overlapFac=0.5, window=np.hanning):
    win = window(frameSize)
    hopSize = int(frameSize - np.floor(overlapFac * frameSize))
    # zeros at beginning (thus center of 1st window should be for sample nr. 0)
    samples = np.append(np.zeros(np.floor(frameSize/2.0)), sig)
    # cols for windowing
    cols = np.ceil( (len(samples) - frameSize) / float(hopSize)) + 1
    # zeros at end (thus samples can be fully covered by frames)
    samples = np.append(samples, np.zeros(frameSize))

    frames = stride_tricks.as_strided(samples, shape=(cols, frameSize), strides=(samples.strides[0]*hopSize, samples.strides[0])).copy()
    frames *= win

    return np.fft.rfft(frames)

""" scale frequency axis logarithmically """
def logscale_spec(spec, sr=44100, factor=20.):
    timebins, freqbins = np.shape(spec)

    scale = np.linspace(0, 1, freqbins) ** factor
    scale *= (freqbins-1)/max(scale)
    scale = np.unique(np.round(scale))

    # create spectrogram with new freq bins
    newspec = np.complex128(np.zeros([timebins, len(scale)]))
    for i in range(0, len(scale)):
        if i == len(scale)-1:
            newspec[:,i] = np.sum(spec[:,scale[i]:], axis=1)
        else:
            newspec[:,i] = np.sum(spec[:,scale[i]:scale[i+1]], axis=1)

    # list center freq of bins
    allfreqs = np.abs(np.fft.fftfreq(freqbins*2, 1./sr)[:freqbins+1])
    freqs = []
    for i in range(0, len(scale)):
        if i == len(scale)-1:
            freqs += [np.mean(allfreqs[scale[i]:])]
        else:
            freqs += [np.mean(allfreqs[scale[i]:scale[i+1]])]

    return newspec, freqs

""" plot spectrogram"""
def plotstft(audiopath, binsize=2**10, png_name='tmp.png', save_png=False, offset=0):
    samplerate, samples = wav.read(audiopath)
    s = stft(samples, binsize)

    sshow, freq = logscale_spec(s, factor=1, sr=samplerate)
    ims = 20.*np.log10(np.abs(sshow)/10e-6) # amplitude to decibel
    timebins, freqbins = np.shape(ims)

    ims = np.transpose(ims)
    ims = np.flipud(ims) # weird - dig into why it needs to be flipped for PIL as opposed to not flipped for matplotlib

    if save_png:
        create_png(ims, png_name)

    return ims

def create_png(im_matrix, png_name):
    image = Image.fromarray(im_matrix)
    image = image.convert('L') # convert to grayscale
    image.save(png_name)


''' Iterate throught wave files and save spectrograms '''
if __name__ == '__main__':
    rootdir = '/Users/ky/Desktop/depression-detect/data/interim'

    for subdir, dirs, files in os.walk(rootdir):
        for file in files:
            if file.endswith('.wav'):
                wav_file = os.path.join(subdir, file)
                png_name = subdir + '/' + file[:-4] + '.png'
                print 'Processing ' + file + '...'
                plotstft(wav_file, png_name=png_name, save_png=True)
