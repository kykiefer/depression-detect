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

cnn_3_final
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
