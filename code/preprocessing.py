import pandas as pd
from sklearn.model_selection import train_test_split

# models
from sklearn.ensemble import RandomForestClassifier


df = pd.read_csv('/Users/ky/Desktop/depression-detect/raw_data/train.csv')

y = df.pop('labels')
X = df

# perform stratifid test train split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=15, stratify=y)


# random forest
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
print('rf train score', rf.score(X_train, y_train))
print('rf test score', rf.score(X_test, y_test))
