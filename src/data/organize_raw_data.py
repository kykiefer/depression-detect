import os
import zipfile
import fnmatch


def extract_files(zip_file, out_dir, delete_zip=False):
    """
    A fucntion takes in a zip file and extracts the .wav file and transcript.csv into separate folders in the current working directory.

    Parameters
    ----------
    filename : sting
        the input zip file

    Returns
    -------
    Two directories. One named 'audio', containing the extracted wav files and one named 'transcripts' containing the extracted transcript csv files.
    """
    # create audio directory
    audio_dir = os.path.join(out_dir, 'audio')
    if not os.path.exists(audio_dir):
        os.makedirs(audio_dir)

    # create transcripts directory
    transcripts_dir = os.path.join(out_dir, 'transcripts')
    if not os.path.exists(audio_dir):
        os.makedirs(transcripts_dir)

    os.chdir(audio_dir)

    # save relevant zip files into relevant directory
    zip_ref = zipfile.ZipFile(zip_file)
    for f in zip_ref.namelist(): # iterates through files in zip file
        if f.endswith('.wav'):
            zip_ref.extract(f, audio_dir)
        elif fnmatch.fnmatch(f, '*TRANSCRIPT.csv'):
            zip_ref.extract(f, transcripts_dir)
    zip_ref.close()

    # remove zip files
    if delete_zip:
        os.remove(zip_file) # delete zip file

if __name__ == '__main__':
    dir_name = '/Volumes/Seagate Backup Plus Drive/DAIC-WOZ/' # directory containing DIAC-WOZ zip files
    extension = ".zip"
    out_dir = '/Users/ky/Desktop/depression-detect/raw_data'

    for file in os.listdir(dir_name):
        if file.endswith(extension):
            filename = os.path.join(dir_name, file)
            extract_files(filename, out_dir, delete_zip=False)
