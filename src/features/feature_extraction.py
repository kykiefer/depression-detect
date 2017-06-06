from pyAudioAnalysis import audioBasicIO as aIO
from pyAudioAnalysis import audioFeatureExtraction as aF
import os


"""
This script iterates through the wav files provided and extracts short-term audio features as numpy matrices of shape (numOfFeatures x numOfShortTermWindows). pyAudioAnalysis provides 34 features (rows) for extraction.
"""

def st_feature_extraction(filename, frame_size=50, step=25):
    """
    A function that implements pyAudioAnalysis' short-term feature extraction.

    Parameters
    ----------
    filename : string
        the input wav file
    frame_size : int
        desired frame size in milliseconds
    step : int
        desired frame step in milliseconds

    Returns
    -------
    st_features : numpy array
        array of numOfFeatures x numOfShortTermWindows
    """
    [Fs, x] = aIO.readAudioFile(filename) # get [sample freq, signal]
    st_features = aF.stFeatureExtraction(x, Fs, frame_size/1000.*Fs, step/1000.*Fs)
    return st_features
