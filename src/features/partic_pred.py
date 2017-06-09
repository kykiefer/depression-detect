from keras.models import load_model

def aggregate_preds(model, X_train, X_test, y_train, y_test):
    y_test_pred = model.predict_classes(X_test)
    y_train_pred = model.predict_classes(X_train)

    y_test_pred_proba = model.predict_proba(X_test)
    y_train_pred_proba = model.predict_proba(X_train)

    # Converting y_test back to 1-D array for confusion matrix computation
    y_test_1d = y_test[:,1]

    # Computing confusion matrix for test dataset
    conf_matrix = confusion_matrix(y_test_1d, y_test_pred)
    print("Confusion Matrix:")
    print(conf_matrix)

    return y_train_pred, y_test_pred, y_train_pred_proba, y_test_pred_proba, conf_matrix

if __name__ == '__main__':
    model = '/Users/ky/Desktop/depression-detect/src/models/cnn_3_final.h5'
    model = load_model(model)
