Baseline architecture
batch_size = 8
epochs = 20
```python
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

cnn_2 (first model with no leaking)
7x7 filter
0.56 accuracy after about 10 epochs


cnn_3
3x3 filter
