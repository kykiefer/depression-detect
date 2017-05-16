# organize raw data --> extracts wav files and transcript csv
# segmentation --> removes silence from clips
# build ml inputs
    # feature extraction --> turns audio segments in to short term feature matrices

"""
This script takes the zipfiles and runs through data organization, silence removal (semgentation), feature extraction, building model inputs, through model validation. It is meant to be used as a script to trace the workflow and for one-stop shop for parameter/hyperparemeter tuning.
"""

import organize_raw data
# starting with a folder of all the zip files from the DAIC-WOZ database extract the wav files and transcript csvs

dir_name = '/Volumes/Seagate Backup Plus Drive/DAIC-WOZ/' # directory containing DIAC-WOZ zip files
