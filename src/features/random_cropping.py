from class_imbalance import get_class_imbalance
import os
from dataframes import df_dev
from PIL import Image

"""
There exists a large data imbalance between positive and negative samples, which incurs a large bias in classification. The number of non-depressed subjects is about four times bigger than that of depressed ones. If these samples for learning, the model will have a strong bias to the non-depressed class. Moreover, regarding the length of each sample, a much longer signal of an individual may emphasize some characteristics that are person specific.

To solve the problem, I perform random cropping on each of the participant's spectrograms of a specified width (time) and constant height (frequency), to ensure the CNN has an equal proportion for every subject and each class.

The size of the Hanning window is 1024, and the audio sample rate is 16000 Hz, which leads to a covering domain of 1024/16000 Hz=0.064s; accordingly, the hop size is 32ms, half of the analysis window. Meaning each pixel of width represents 32ms.

Some success has been found using random crops of 3-5 seconds [reference]. I'll start with 4 seconds and tune it like a hyperparameter. A width of 4 seconds means 4/0.032 = 125 pixel window size.

Reference: DeepAudioNet: An Efficient Deep Model for Audio based Depression Classification
"""

crop_width = 125 # 125 pixels = 4 seconds of audio

def deterimine_how_many_samples_to_crop_from_each_particpant(depressed_dict, normal_dict):
    print(depressed_dict)
    print(normal_dict)


def randomly_crop_n_unique_samples_from_each_participant(n):
    pass

def crop_image(png_file, out_png):
    print png_file
    print out_png
    # png = '/Users/ky/Desktop/depression-detect/data/interim/P301/P301_no_silence.png'
    # img = Image.open(png)
    # width, height = img.size
    # img2 = img.crop((0, 0, width, height))
    # img2.save("img2.png")

if __name__ == '__main__':
    # directory containing participant's folders (which contains the no_silence.png)
    rootdir = '/Users/ky/Desktop/depression-detect/data/interim'
    depressed_dict, normal_dict = get_class_imbalance(rootdir)
    deterimine_how_many_samples_to_crop_from_each_particpant(depressed_dict, normal_dict)
