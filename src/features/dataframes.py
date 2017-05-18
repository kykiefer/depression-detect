import pandas as pd

df_train = pd.read_csv('/Users/ky/Desktop/depression-detect/data/raw/labels/train_split_Depression_AVEC2017.csv')
df_test = pd.read_csv('/Users/ky/Desktop/depression-detect/data/raw/labels/dev_split_Depression_AVEC2017.csv')

# making a decision to combine the pre identified test and train sets (all having depression labels) to ensure stratified and balanced classes with audio segmentation (silence and speaker removal) challenges
df_dev = pd.concat([df_train, df_test], axis=0)

# df_holdout = pd.read_csv('/Users/ky/Desktop/depression-detect/raw_data/labels/dev_split_Depression_AVEC2017.csv')
