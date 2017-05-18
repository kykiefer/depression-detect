from class_imbalance import get_class_imbalance
import os
from dataframes import df_dev
from PIL import Image

"""
There exists a large data imbalance between positive and negative samples, which incurs a large bias in classification. The number of non-depressed subjects is about four times bigger than that of depressed ones. If these samples for learning, the model will have a strong bias to the non-depressed class. Moreover, regarding the length of each sample, a much longer signal of an individual may emphasize some characteristics that are person specific.

To solve the problem, I perform random cropping on each of the particpant's spectrograms of a specified width (time) and contstant height (frquency -- the whole spectrum), to ensure the CNN has an equal proportion for every subject and each class.
"""

def crop_image(png_file, out_png):
    print png_file
    print out_png
    # png = '/Users/ky/Desktop/depression-detect/data/interim/P301/P301_no_silence.png'
    # img = Image.open(png)
    # width, height = img.size
    # img2 = img.crop((0, 0, width, height))
    # img2.save("img2.png")
