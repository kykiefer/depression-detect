import os
from dataframes import df_dev
from PIL import Image

"""
There exists a large data imbalance between positive and negative samples, which incurs a large bias in classification. The number of non-depressed subjects is about four times bigger than that of depressed ones. If these samples for learning, the model will have a strong bias to the non-depressed class. Moreover, regarding the length of each sample, a much longer signal of an individual may emphasize some characteristics that are person specific.

To solve the problem, I perform random cropping on each of the particpant's spectrograms of a specified width (time) and contstant height (frquency -- the whole spectrum), to ensure the CNN has an equal proportion for every subject and each class.
"""

def get_class_imbalance(rootdir):
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
    return df_dev.loc[df_dev['Participant_ID'] == partic_id]['PHQ8_Binary'].item()

def in_dev_split(partic_id):
    return partic_id in set(df_dev['Participant_ID'].values)

if __name__ == '__main__':
    rootdir = '/Users/ky/Desktop/depression-detect/data/interim'
    depressed_dict, normal_dict = get_class_imbalance(rootdir)
    print(depressed_dict)
    print(normal_dict)
