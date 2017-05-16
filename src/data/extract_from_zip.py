import fnmatch
import os
import zipfile


"""
A script iterates through a directory of the 189 DAIC-WOZ partcipiant zip files and extracts the wav and transcipt files relavant to analysis.
"""

def extract_files(zip_file, out_dir, delete_zip=False):
    """
    A fucntion takes in a zip file and extracts the .wav file and *TRANSCRIPT.csv files into separate folders in the a user specified directory.

    Parameters
    ----------
    zip_file : string
        path to the input zip file
    out_dir : string
        path to the desired directory (where audio and transcipt folders will be created)
    delete_zip : bool
        If true, deletes the zip file onces relevant files are extracted

    Returns
    -------
    Two directories :
        audio: containing the extracted wav files
        transcripts: containing the extracted transcript csv files
    """
    # create audio directory
    audio_dir = os.path.join(out_dir, 'audio')
    if not os.path.exists(audio_dir):
        os.makedirs(audio_dir)

    # create transcripts directory
    transcripts_dir = os.path.join(out_dir, 'transcripts')
    if not os.path.exists(audio_dir):
        os.makedirs(transcripts_dir)

    # save relevant unzipped files into appropriate directory
    zip_ref = zipfile.ZipFile(zip_file)
    for f in zip_ref.namelist(): # iterates through files in zip file
        if f.endswith('.wav'):
            zip_ref.extract(f, audio_dir)
        elif fnmatch.fnmatch(f, '*TRANSCRIPT.csv'):
            zip_ref.extract(f, transcripts_dir)
    zip_ref.close()

    if delete_zip:
        os.remove(zip_file)


if __name__ == '__main__':
    dir_name = '/Volumes/Seagate Backup Plus Drive/DAIC-WOZ/' # directory containing DIAC-WOZ zip files
    extension = '.zip'
    out_dir = '/Users/ky/Desktop/depression-detect/data/raw'

    for file in os.listdir(dir_name):
        if file.endswith(extension):
            zip_file = os.path.join(dir_name, file)
            extract_files(zip_file, out_dir, delete_zip=False)
