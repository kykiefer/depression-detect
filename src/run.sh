# organize raw data --> extracts wav files and transcript csv
# segmentation --> removes silence from clips
# build ml inputs
    # feature extraction --> turns audio segments in to short term feature matrices

"""
This script takes the zipfiles and runs through data organization, silence removal (semgentation), feature extraction, building model inputs, through model validation. It is meant to be used as a script to trace the workflow and for one-stop shop for parameter/hyperparemeter tuning.
"""

# starting with a folder of all the zip files from the DAIC-WOZ database extract the wav files and transcript csvs

# starting with a folder of all the zip files from the DAIC-WOZ database extract the wav files and transcript csvs
python organize_raw_data.py

# segmentation removes silence from clips and and saves the segmented wav files
python segmentation.py

# builds a feature matrix from the segmented wav files
python build_ml_inputs.py
