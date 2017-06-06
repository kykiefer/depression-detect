from audio_feature_dict import st_feat_dict
from dataframes import df_dev
import feature_extraction as fE
import numpy as np
import os
import pandas as pd


"""
This script assembles the audio features and associated depression labels in a format that is consumable by various machine learning algorithms.
"""

def assemble_dfs(dir_name, extension='.wav'):
    """
    Iterates through audio segments and builds a dataframe with the short term features for every frame in a segment and with an associated depression label.
    """
    final_df_ls = []
    for subdir, dirs, files in os.walk(dir_name):
        for file in files:
            if file.endswith('no_silence.wav'):
                filename = os.path.join(dir_name, subdir, file)
                partic_id = int(filename.split('/')[-1].split('_')[0][1:])
                if in_test_train(partic_id):
                    print(filename)
                    df = build_df(filename, partic_id) # build df for segment
                    final_df_ls.append(df)
    final_df = pd.concat(final_df_ls, axis=0)
    return final_df


def build_df(filename, partic_id):
    """
    Builds short-term feature array of shape (numOfFeatures x numFrames). Transposes it so each row in the dataframe is a frame of 34 feature with the last column being the assoicated deperession label for that particpant.
    """
    st_feat_array = fE.st_feature_extraction(filename)
    df = pd.DataFrame(st_feat_array.T, columns=st_feat_dict.values())
    depression_label = get_deperession_label(partic_id)
    labels = np.ones((df.shape[0], 1)) * depression_label
    df['target'] = labels.astype(int)
    return df


def in_test_train(partic_id):
    """
    Verifies wav file is in test train split (i.e. has depression label), as opposed to the holdout DAIC-WOZ dataset which labels were not released for.
    """
    return partic_id in set(df_dev['Participant_ID'])


def get_deperession_label(partic_id):
    """
    Get particpants' depression labels.
    """
    binary_label = df_dev.loc[df_dev['Participant_ID'] == partic_id]['PHQ8_Binary'].values[0] # 1 is depression, 0 is not
    return binary_label


if __name__ == '__main__':
    dir_name = '/Users/ky/Desktop/depression-detect/data/interim/' # directory containing segmented wav files

    # build pandas df
    final_df = assemble_dfs(dir_name)

    # write to csv
    out_dir = '/Users/ky/Desktop/depression-detect/data/processed'
    final_df.to_csv(os.path.join(out_dir, 'test_train.csv'), index=False)
