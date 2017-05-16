import pandas as pd

df_train = pd.read_csv('/Users/ky/Desktop/depression-detect/data/raw/labels/train_split_Depression_AVEC2017.csv')
df_test = pd.read_csv('/Users/ky/Desktop/depression-detect/data/raw/labels/dev_split_Depression_AVEC2017.csv')

df_test_train = pd.concat([df_train, df_test], axis=0)

# df_holdout = pd.read_csv('/Users/ky/Desktop/depression-detect/raw_data/labels/dev_split_Depression_AVEC2017.csv')
