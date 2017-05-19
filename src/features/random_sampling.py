from spectrogram_dicts import build_class_dictionaries
from cnn_spectrograms import stft_matrix
import numpy as np
import os
from dataframes import df_dev
from PIL import Image

"""
There exists a large data imbalance between positive and negative samples, which incurs a large bias in classification. The number of non-depressed subjects is about four times bigger than that of depressed ones. If these samples for learning, the model will have a strong bias to the non-depressed class. Moreover, regarding the length of each sample, a much longer signal of an individual may emphasize some characteristics that are person specific.

To solve the problem, I perform random cropping on each of the participant's spectrograms of a specified width (time) and constant height (frequency), to ensure the CNN has an equal proportion for every subject and each class.

The size of the Hanning window is 1024, and the audio sample rate is 16000 Hz, which leads to a covering domain of 1024/16000 Hz=0.064s; accordingly, the hop size is 32ms, half of the analysis window. Meaning each pixel of width represents 32ms.

Some success has been found using random crops of 3-5 seconds [1]. I'll start with 4 seconds and tune it like a hyperparameter. A width of 4 seconds means 4/0.032 = 125 pixel window size.

[1] DeepAudioNet: An Efficient Deep Model for Audio based Depression Classification
"""


def determine_num_crops(depressed_dict, crop_width=125):
    """
    Finds the shortest clip in the depressed class, which, according to our random sampling strategy, will limit the number of samples we take from each clip to make sure our classes are balanced.

    Parameters
    ----------
    depressed_dict : dictionary
        a dictionary of depressed particpants with the participant id as the key and the segmented and concatentated matrix representation of their spectrograms as the values.
    crop_width : integer
        the desired pixel width of the crop samples (125 pixes = 4 seconds of audio)

    Returns
    -------
    num_samples_from_clips : int
        the maximum number of samples that should be sampled from each clip to ensure balanced classes can be built
    """
    shortest_clip = min(depressed_dict.items(), key=lambda x: x[1].shape[1]) # returns dictionary entry
    shortest_pixel_width = shortest_clip[1].shape[1]
    num_samples_from_clips = shortest_pixel_width / crop_width
    return num_samples_from_clips

def get_samples_from_class(segmented_audio_dict, n_samples, crop_width):
    """
    Get N (num_samples) pseudo random non-overlapping samples from the all the depressed participants.

    Parameters
    ----------
    segmented_audio_dict : dictionary
        a dictionary of a class of particpants with keys of participant ids and values of the segmented audio matrix spectrogram representation
    n_samples : integer
        number of pseudo-random non-overlapping samples to extract from each segmented audio matrix spectrogram
    crop_width : integer
        the desired pixel width of the crop samples (125 pixes = 4 seconds of audio)

    Returns
    -------
    class sample dict : dictionary
        a dictionary of a class of particpants with keys of participant ids and values of a list of the croped samples from the specgrogram matrices. The lists are n_samples long and the entries within the list have dimension (numFrequencyBins * crop_width)
    """
    class_samples_dict = dict()
    for partic_id, clip_mat in segmented_audio_dict.iteritems():
            # print(partic_id)
            samples = random_non_overlapping_samples(clip_mat, n_samples, crop_width)
            class_samples_dict[partic_id] = samples
    return class_samples_dict

def random_non_overlapping_samples(matrix, n_samples, crop_width):
    """
    Get N pseudo-random samples with width of crop_width from the numpy matrix representing the partiipants audio spectrogram.
    """
    width = matrix.shape[1]
    freedom = width - (n_samples * crop_width) # total width - width to be sampled
    # print(width)
    # print('{} samples with {} crop width'.format(n_samples, crop_width))
    # print('{} pixels of freedom'.format(freedom))

    partic_samples = []
    start_col = 0
    for sample in range(n_samples):
        offset = np.random.randint(0, freedom) # randomness gets eaten up pretty quickly -- come back to this...maybe okay
        freedom -= offset # remaining freedom
        start_col += offset
        end_col = start_col + crop_width
        crop = matrix[:, start_col:end_col] # all frquency bins; specified time slice
        partic_samples.append(crop)
        start_col += crop_width
    return partic_samples

def create_sample_dicts():
    """
    Utilizes the above function to return two dictionaries, depressed and normal. Each dictionary has only participants in the specific class, with participant ids as key, a values of a list of the croped samples from the spectorgram matrices. The lists are n_samples long and the entries within the list have dimension
    """
    np.random.seed(15) # for reproducibility
    crop_width = 125 # 125 pixels = 4 seconds of audio
    # build dictonaries of participants and segmented audio matrix
    depressed_dict, normal_dict = build_class_dictionaries('/Users/ky/Desktop/depression-detect/data/interim')
    n_samples = determine_num_crops(depressed_dict, crop_width=crop_width)
    # get n_sample pseudo-random samples from each depressed participant
    depressed_samples = get_samples_from_class(depressed_dict, n_samples, crop_width)
    # get n_sample pseudo-random samples from each non-depressed participant
    normal_samples = get_samples_from_class(normal_dict, n_samples, crop_width)
    return depressed_samples, normal_samples
