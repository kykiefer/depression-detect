import os
from dataframes import df_dev
from PIL import Image


def get_class_imbalance(rootdir):
    """
    Using the width (represeting time) time of each particpant's no_silence.png spectrogram, return a dictionary with keys of particpants and values of png width. This can be used to quantify how much audio/spectrogram is available for each class (depressed/non-depressed) and each speaker (particpant)

    Parameters
    ----------
    rootdir : filepath
        directory containing participant's folders (which contains the no_silence.png)
    Returns
    -------
    depressed_dict : dictionary
        dictionary of depressed individuals with keys of particiapnt id and values of no_silence spectrogram width
    normal_dict : dictionary
        dictionary of non-depressed individuals with keys of particiapnt id and values of no_silence spectrogram width
    """
    depressed_dict = dict()
    normal_dict = dict()
    for subdir, dirs, files in os.walk(rootdir):
        for file in files:
            if file.endswith('no_silence.png'):
                partic_id = int(file.split('_')[0][1:])
                if in_dev_split(partic_id):
                    png_file = os.path.join(subdir, file)
                    width = get_weight(png_file) # representative of time
                    depressed = get_depression_label(partic_id) # 1 if True
                    if depressed:
                        depressed_dict[partic_id] = width
                    elif not depressed:
                        normal_dict[partic_id] = width
    return depressed_dict, normal_dict

def get_weight(png_file):
    """
    Returns the width of the spectrogram, which corresponds to the length of the clips.
    """
    img = Image.open(png_file)
    width = img.size[0]
    return width

def get_depression_label(partic_id):
    """
    Returns participant's PHQ8 Binary label. 1 representing depression; 0 representing no depression.
    """
    return df_dev.loc[df_dev['Participant_ID'] == partic_id]['PHQ8_Binary'].item()

def in_dev_split(partic_id):
    """
    Returns True if the participant is in the AVEC developemnt split (aka particpant's we have depression labels for)
    """
    return partic_id in set(df_dev['Participant_ID'].values)

if __name__ == '__main__':
    rootdir = '/Users/ky/Desktop/depression-detect/data/interim'
    depressed_dict, normal_dict = get_class_imbalance(rootdir)
    print(depressed_dict)
    print(normal_dict)
