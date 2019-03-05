import sys
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
import keras

import time

start = time.time()

model_path = 'my_image_recognizerr.model'

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

if 'load' in sys.argv:
    print("loading model from: ", model_path)
    model = keras.models.load_model(model_path)
else:
    print("training new model")
    # normalize input
    x_train = keras.utils.normalize(x_train, axis=1)
    x_test = keras.utils.normalize(x_test, axis=1)

    model = keras.models.Sequential()
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(128, activation=tf.nn.relu))
    model.add(keras.layers.Dense(128, activation=tf.nn.relu))
    model.add(keras.layers.Dense(10, activation=tf.nn.softmax))

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    model.fit(x_train, y_train, epochs=3)

    # test the model
    val_loss, val_acc = model.evaluate(x_test, y_test)
    print("Loss on test set: ", val_loss)
    print("Accuracy on test set: ", val_acc)

    if 'save' in sys.argv:
        print("saving model into: ", model_path)
        keras.models.save_model(model=model, filepath=model_path)
    else:
        print("new trained model will not be saved")

end = time.time()

print("Elapsed time: ", end-start)

input_index = 7
predictions = model.predict([x_test])
# print(predictions)
print("predicted number is: ", np.argmax(predictions[input_index]))

plt.imshow(x_test[input_index], cmap=plt.cm.binary)
plt.show()
