import os
import feature_extraction as fE
import pandas as pd
from audio_feature_dict import st_feat_dict

"""
This script assembles the audio features and associated depression labels in a format that is consumable by various machine learning algorithms.
"""
def assemble_dfs(dir_name, extension):
    final_df_ls = []
    for file in os.listdir(dir_name): # for audio segment
        if file.endswith(extension):
            filename = os.path.join(dir_name, file)
            df = build_df(filename) # build df for segment
            final_df_ls.append(df)
    final_df = pd.concat(final_df_ls, axis=0)
    return final_df

def build_df(filename):
    st_feat_array = fE.st_feature_extraction(filename)
    depression_label = get_deperession_label(filename) # WORK ON THIS -- use temp.py
    df = pd.DataFrame(st_feat_array.T, columns=st_feat_dict.values())
    return df

def get_deperession_label(filename):
    """
    Get particpants' depressed labels.
    """
    partic_id = filename.split('/')[-1].split('_')[0]
    print(partic_id)

if __name__ == '__main__':
    dir_name = '/Users/ky/Desktop/depression-detect/raw_data/audio_segments' # directory containing segmented wav files
    extension = ".wav"
    out_dir = '/Users/ky/Desktop/depression-detect/raw_data'

    final_df = assemble_dfs(dir_name, extension)
