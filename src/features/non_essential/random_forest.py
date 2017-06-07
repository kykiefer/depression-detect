import pandas as pd
import pickle
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score


def run_forest(X_train, X_test, y_train, y_test):
    print('Training Random Forest...')
    rf = RandomForestClassifier(n_estimators=25, criterion='gini', random_state=15)
    rf.fit(X_train, y_train)

    # print('Train accuracy: {}'.format(rf.score(X_train, y_train)))
    # print('Test accuracy: {}'.format(rf.score(X_test, y_test)))
    #
    # y_pred_train = rf.predict(X_train)
    # print('Train f1: {}'.format(f1_score(y_train, y_pred_train)))
    # y_pred_test = rf.predict(X_test)
    # print('Test f1: {}'.format(f1_score(y_test, y_pred_test)))

    print('Cross val scores:')
    print('accuracy', np.mean(cross_val_score(rf, X_train, y_train, scoring='accuracy')))
    print('precision', np.mean(cross_val_score(rf, X_train, y_train, scoring='precision')))
    print('recall', np.mean(cross_val_score(rf, X_train, y_train, scoring='recall')))
    print('f1', np.mean(cross_val_score(rf, X_train, y_train, scoring='f1')))

    print('feature importances', rf.feature_importances_)

    # test prediciton
    y_test_pred = rf.predict(X_test)
    print('f1 test', f1_score(y_test, y_test_pred))

    return rf


if __name__ == '__main__':
    df = pd.read_csv('/Users/ky/Desktop/depression-detect/data/processed/test_train.csv')

    # under sample for even classes
    # fraud = df.loc[df.fraud == 1]
    # n_samps = fraud.shape[0]
    # not_fraud = df.loc[df.fraud == 0].sample(n_samps, random_state=15)
    # df = pd.concat([fraud, not_fraud])
    n_samps = 40000
    depressed = df.loc[df.target == 1].sample(n_samps, random_state=15)
    normal = df.loc[df.target == 0].sample(n_samps, random_state=15)
    df = pd.concat([depressed, normal])

    # create features and target
    y = df.pop('target')
    X = df

    # train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=15)

    print('Training on {} samples'.format(X_train.shape[0]))
    print('Testing on {} samples'.format(X_test.shape[0]))

    model = run_forest(X_train, X_test, y_train, y_test)

    with open('/Users/ky/Desktop/depression-detect/src/models/rf.pkl', 'wb') as f:
        pickle.dump(model, f)
