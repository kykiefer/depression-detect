import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

# models
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

df = pd.read_csv('/Users/ky/Desktop/depression-detect/raw_data/train.csv')
y = df.pop('labels')
X = df

# perform stratifid test train split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=15, stratify=y)

# # random forest
# forest = RandomForestClassifier(n_estimators=10, random_state=15)
# forest.fit(X_train, y_train)
#
# # predict and score
# y_pred_train = forest.predict(X_train)
# y_pred_test = forest.predict(X_test)
# print('rf train f1 score:', f1_score(y_train, y_pred_train))
# print('rf train test score:', f1_score(y_test, y_pred_test))
# importances = forest.feature_importances_
# print('feature importances:', importances)

# boosted forest
boosted_forest = GradientBoostingClassifier(max_depth=3)
boosted_forest.fit(X_train, y_train)

# predict and score
y_pred_train = boosted_forest.predict(X_train)
y_pred_test = boosted_forest.predict(X_test)
print('rf train f1 score:', f1_score(y_train, y_pred_train))
print('rf train test score:', f1_score(y_test, y_pred_test))
