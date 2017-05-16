import os
import feature_extraction as fE
import pandas as pd
import numpy as np
from audio_feature_dict import st_feat_dict
from dataframes import df_participants

"""
This script assembles the audio features and associated depression labels in a format that is consumable by various machine learning algorithms.
"""

def assemble_dfs(dir_name, extension):
    """
    Iterates through audio segments and builds a dataframe with the short term features for every frame in a segment and with an associated depression label.
    """
    final_df_ls = []
    for file in os.listdir(dir_name): # for audio segment
        print(file) # temp
        if file.endswith(extension):
            filename = os.path.join(dir_name, file)
            df = build_df(filename) # build df for segment
            final_df_ls.append(df)
    final_df = pd.concat(final_df_ls, axis=0)
    final_df.to_csv('train.csv', index=False)
    return final_df

def build_df(filename):
    """
    Builds short-term feature array of shape (numOfFeatures x numFrames). Transposes it so each row in the dataframe is a frame with an associated depression label.
    """
    st_feat_array = fE.st_feature_extraction(filename)
    df = pd.DataFrame(st_feat_array.T, columns=st_feat_dict.values())
    depression_label = get_deperession_label(filename) # WORK ON THIS -- use temp.py
    labels = np.ones((df.shape[0], 1)) * depression_label
    df['labels'] = labels.astype(int)
    return df

def get_deperession_label(filename):
    """
    Get particpants' depression labels.
    """
    partic_id = int(filename.split('/')[-1].split('_')[0][1:])
    binary_label = df_participants.loc[df_participants['Participant_ID'] == partic_id]['PHQ8_Binary'].values[0] # 1 is depression, 0 is not
    return binary_label

if __name__ == '__main__':
    dir_name = '/Users/ky/Desktop/depression-detect/raw_data/audio_segments' # directory containing segmented wav files
    extension = ".wav"
    out_dir = '/Users/ky/Desktop/depression-detect/raw_data'

    final_df = assemble_dfs(dir_name, extension)
