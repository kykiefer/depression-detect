from spectrograms import stft_matrix
import os
from dev_data import df_dev


"""
This script builds dictionaries for the depressed and non-depressed classes
with each participant id as the key, and the associated segmented matrix
spectrogram representation as the value. Said values can than be cropped and
randomly sampled as input to the CNN.
"""


def build_class_dictionaries(dir_name):
    """
    Builds a dictionary of depressed participants and non-depressed
    participants with the participant id as the key and the matrix
    representation of the no_silence wav file as the value. These
    values of this dictionary are then randomly cropped and sampled
    from to create balanced class and speaker inputs to the CNN.
    Parameters
    ----------
    dir_name : filepath
        directory containing participant's folders (which contains the
        no_silence.wav)
    Returns
    -------
    depressed_dict : dictionary
        dictionary of depressed individuals with keys of participant id
        and values of with the matrix spectrogram representation
    normal_dict : dictionary
        dictionary of non-depressed individuals with keys of participant id
        and values of with the matrix spectrogram representation
    """
    depressed_dict = dict()
    normal_dict = dict()
    for subdir, dirs, files in os.walk(dir_name):
        for file in files:
            if file.endswith('no_silence.wav'):
                partic_id = int(file.split('_')[0][1:])
                if in_dev_split(partic_id):
                    wav_file = os.path.join(subdir, file)
                    # matrix representation of spectrogram
                    mat = stft_matrix(wav_file)
                    depressed = get_depression_label(partic_id)  # 1 if True
                    if depressed:
                        depressed_dict[partic_id] = mat
                    elif not depressed:
                        normal_dict[partic_id] = mat
    return depressed_dict, normal_dict


def in_dev_split(partic_id):
    """
    Returns True if the participant is in the AVEC development split
    (aka participant's we have depression labels for)
    """
    return partic_id in set(df_dev['Participant_ID'].values)


def get_depression_label(partic_id):
    """
    Returns participant's PHQ8 Binary label. 1 representing depression;
    0 representing no depression.
    """
    return df_dev.loc[df_dev['Participant_ID'] ==
                      partic_id]['PHQ8_Binary'].item()


if __name__ == '__main__':
    dir_name = '../../data/interim'
    depressed_dict, normal_dict = build_class_dictionaries(dir_name)
