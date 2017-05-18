import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_curve, auc
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC

df = pd.read_csv('/Users/ky/Desktop/depression-detect/data/processed/test_train.csv')
# df = df.sample(n=1000) # subsample
y = df.pop('target')
X = df

# perform stratified test train split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=15, stratify=y)

def random_forest(X_train, X_test, y_train, y_test):
    forest = RandomForestClassifier(n_estimators=10, random_state=15, class_weight='balanced')
    forest.fit(X_train, y_train)

    f1 = cross_val_score(forest, X_train, y_train, cv=5, scoring='f1')
    print('F1 Score: {0:.2f} (+/- {1:.2f})'.format(f1.mean(), f1.std() * 2))
    print('F1 Train:', f1_score(y_train, forest.predict(X_train)))

    auc = cross_val_score(forest, X_train, y_train, cv=5, scoring='roc_auc')
    print('AUC Score: {0:.2f} (+/- {1:.2f})'.format(auc.mean(), auc.std() * 2))

    # other random forest stats
    # importances = forest.feature_importances_
    # print('feature importances:', importances)

def support_vector_classifier(X_train, X_test, y_train, y_test):
    support_clf = SVC(C=1.0, kernel='rbf', class_weight='balanced', random_state=15)
    support_clf.fit(X_train, y_train)

    print('score', support_clf.score(X_test, y_test))

    f1 = cross_val_score(support_clf, X_train, y_train, cv=5, scoring='f1')
    print('F1 Score: {0:.2f} (+/- {1:.2f})'.format(f1.mean(), f1.std() * 2))
    print('F1 Train:', f1_score(y_train, support_clf.predict(X_train)))

    auc = cross_val_score(support_clf, X_train, y_train, cv=5, scoring='roc_auc')
    print('AUC Score: {0:.2f} (+/- {1:.2f})'.format(auc.mean(), auc.std() * 2))

if __name__ == '__main__':
    print('Random Forest')
    random_forest(X_train, X_test, y_train, y_test)
    print('SVC')
    support_vector_classifier(X_train, X_test, y_train, y_test)
