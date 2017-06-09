Baseline architecture

```python
batch_size = 8
epochs = 20
model = Sequential()

model.add(Conv2D(32, (7, 7), padding='valid', strides=1, input_shape=input_shape, activation='relu', kernel_initializer='random_uniform'))
model.add(MaxPooling2D(pool_size=(4,3), strides=(1,3)))
model.add(Conv2D(32, (1, 3), padding='valid', strides=1, input_shape=input_shape, activation='relu'))
model.add(MaxPooling2D(pool_size=(1,3), strides=(1,3)))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

# sgd = SGD(lr=0.001, decay=1e-6, momentum=1.0)
model.compile(loss='categorical_crossentropy',
              optimizer='adadelta',
              metrics=['accuracy'])

history = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs,
          verbose=1, validation_data=(X_test, y_test))
```

first Conv2D filter size (accuracy)
7x7 0.56 @ 10 epochs - cnn2
3x3 0.58 @ 11 epochs - cnn3
5x5 did not converge - cnn4

cnn_3
3x3 filter
Train accuracy: 0.983467741935
Test accuracy: 0.555357142857
Confusion Matrix:
[[115 165]
 [ 84 196]]
Accuracy: 0.5553571428571429
Precision: 0.4107142857142857
Recall: 0.5778894472361809
F1-Score: 0.4801670146137787
Notes: loss best around 11 epochs, acc ~ 0.58

cnn_5 - NOPE
3x3 filter, changed up pooling dimensions
```python
model.add(Conv2D(32, (3, 3), padding='valid', strides=1, input_shape=input_shape, activation='relu', kernel_initializer='random_uniform'))
model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2)))
model.add(Conv2D(32, (1, 3), padding='valid', strides=1, input_shape=input_shape, activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=(1,1)))
```

*cnn_3_final*
same as cnn_3 but run for 11 epochs to minimize loss
Train accuracy: 0.813709677419
Test accuracy: 0.555357142857
Confusion Matrix:
[[156 124]
 [125 155]]
Accuracy: 0.5553571428571429
Precision: 0.5571428571428572
Recall: 0.5551601423487544
F1-Score: 0.5561497326203209

cnn_6
cnn_3 but with 80 filters in lieu of 32




SUBSAMPLES (HALF)

Batch size
    f1, precision, recall
8 - 0.598, 0.617, 0.580
10 - 0.57
larger batches getting memory errors - moving to c4.8xlarge instance to allow for more nodes

cnn_3_subsample
12 epochs baseline.
Train accuracy: 0.757258064324
Test accuracy: 0.585714285714
Confusion Matrix:
[[173 107]
 [125 155]]
Accuracy: 0.5857142857142857
Precision: 0.6178571428571429
Recall: 0.5805369127516778
F1-Score: 0.5986159169550174

cnn_6sub - did not converge
Added convolution layers
Starting to look like http://yerevann.github.io/2015/10/11/spoken-language-identification-with-deep-convolutional-networks/

cnn8_sub -
cnn_3 but with 200 nodes in dense layers instead of 128

cnn512 (c4.8xlarge)
512 nodes in two dense layers
rain accuracy: 0.663709677419
Test accuracy: 0.535714285714
Confusion Matrix:
[[ 78 202]
 [ 58 222]]
Accuracy: 0.5357142857142857
Precision: 0.2785714285714286
Recall: 0.5735294117647058
F1-Score: 0.375

cnn9_sub - noppe
50% dropout after first pooling and both dense

cnn10_sub - nope
25% dropout after first pooling and both dense

*cnn11_sub* high recall (i.e. the percentage of depressed who are correctly identified as having the condition). bad precision
One dense layer 512 nodes with 25% dropout
Train accuracy: 0.684677419355
Test accuracy: 0.583928571429
Confusion Matrix:
[[104 176]
 [ 57 223]]
Accuracy: 0.5839285714285715
Precision: 0.37142857142857144
Recall: 0.6459627329192547
F1-Score: 0.471655328798186

*cnn12_sub* 11 epochs
One dense layer 512 nodes with 50% dropout
Train accuracy: 0.743548387289
Test accuracy: 0.594642857143
Evaluating model...
Confusion Matrix:
[[152 128]
 [ 99 181]]
Accuracy: 0.5946428571428571
Precision: 0.5428571428571428
Recall: 0.6055776892430279
F1-Score: 0.5725047080979284

cnn13_sub - changed maxpooling shapes 4x4, 3x3
Train accuracy: 0.61209677429
Test accuracy: 0.548214285714
Confusion Matrix:
[[ 58 222]
 [ 31 249]]
Calculating test metrics...
Accuracy: 0.5482142857142858
Precision: 0.20714285714285716
Recall: 0.651685393258427
F1-Score: 0.31436314363143636


cnn14_sub - potentially, didn't converge quite yet more epochs below
Train accuracy: 0.654838709293
Test accuracy: 0.539285714286
Confusion Matrix:
[[124 156]
 [102 178]]
Calculating test metrics...
Accuracy: 0.5392857142857143
Precision: 0.44285714285714284
Recall: 0.5486725663716814
F1-Score: 0.4901185770750988


THESE WERE ALL ON 1/5th of the data
cnn12_full - but only 5 epochs by mistake!
Train accuracy: 0.778225806452
Test accuracy: 0.521428571429
Confusion Matrix:
[[196  84]
 [184  96]]
 Accuracy: 0.5214285714285715
Precision: 0.7
Recall: 0.5157894736842106
F1-Score: 0.593939393939394
Saving plots...
Saving model to S3...

cnn12_final - garbage after 11 epochs


cnn12_20e
Train accuracy: 0.913306451613
Test accuracy: 0.525
Evaluating model...
560/560 [==============================] - 1s
496/496 [==============================] - 1s
560/560 [==============================] - 1s
496/496 [==============================] - 1s
Confusion Matrix:
[[228  52]
 [214  66]]
Saving model locally...
Calculating test metrics...
Accuracy: 0.525
Precision: 0.8142857142857143
Recall: 0.5158371040723982
F1-Score: 0.631578947368421

cnn12_12e
Train accuracy: 0.685483870968
Test accuracy: 0.503571428571
Confusion Matrix:
[[257  23]
 [255  25]]
Accuracy: 0.5035714285714286
Precision: 0.9178571428571428
Recall: 0.501953125
F1-Score: 0.648989898989899

THESE ARE RUN ON THE FULL TEST SET
12_12e.25d_full
Train accuracy: 0.770564516129
Test accuracy: 0.5625
Confusion Matrix:
[[152 128]
 [117 163]]
Accuracy: 0.5625
Precision: 0.5428571428571428
Recall: 0.5650557620817844
F1-Score: 0.5537340619307832

12_9e.5d_full
Train accuracy: 0.700806451613
Test accuracy: 0.571428571429
Confusion Matrix:
[[175 105]
 [135 145]]
Saving model locally...
Calculating test metrics...
Accuracy: 0.5714285714285714
Precision: 0.625
Recall: 0.5645161290322581
F1-Score: 0.5932203389830509

*12_9e.5d_full_32b* best so far
Train accuracy: 0.614516129032
Test accuracy: 0.55
Confusion Matrix:
[[240  40]
 [212  68]]
Accuracy: 0.55
Precision: 0.8571428571428571
Recall: 0.5309734513274337
F1-Score: 0.6557377049180328
