from pyAudioAnalysis import audioBasicIO
from pyAudioAnalysis import audioFeatureExtraction
from audio_feature_dict import st_feat_dict
import matplotlib.pyplot as plt


def st_feature_extraction(filename, frame_size=50, step=25):
    """
    A function that implements pyAudioAnalysis' short-term feature extraction.

    This results to a sequence of feature vectors, stored in a numpy matrix. Descriptions of the 34 features can be found here: https://github.com/tyiannak/pyAudioAnalysis/wiki/3.-Feature-Extraction

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
    [Fs, x] = audioBasicIO.readAudioFile(filename) # get [sample freq, signal]
    st_features = audioFeatureExtraction.stFeatureExtraction(x, Fs, frame_size/1000.*Fs, step/1000.*Fs)
    return st_features

def plot_st_features(st_feat):
    num_feats = st_feat.shape[0]
    print(num_feats)


    # plt.subplot(2,1,1)
    # plt.plot(st_feat[0,:]); plt.xlabel('Frame no')
    # plt.ylabel('ZCR')
    # plt.subplot(2,1,2)
    # plt.plot(st_feat[1,:])
    # plt.xlabel('Frame no')
    # plt.ylabel('Energy')
    # plt.show()

if __name__ == '__main__':
    filename = "sample_clips/P369_happy_clip.wav"
    happy = st_feature_extraction(filename)
    plot_st_features(happy)
