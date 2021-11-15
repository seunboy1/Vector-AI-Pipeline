
import numpy as np
import tensorflow as tf
from tensorflow.keras import datasets, utils
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, MaxPooling2D, Flatten

def mnist_keras():
    mnist = datasets.mnist
    (x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()
    
    # Make sure images have shape (28, 28, 1)
    x_train = np.expand_dims(x_train, axis=3)
    x_test = np.expand_dims(x_test, axis=3)
    
    # Scale images to the [0, 1] range
    x_train = x_train / 255.0
    x_test = x_test / 255.0

    print("x_train shape:", x_train.shape)
    print(x_train.shape[0], "train samples")
    print(x_test.shape[0], "test samples")

    return x_train, x_test, y_train, y_test



def architecture(input_shape=(28, 28, 1), classes=10):
    model = Sequential([
            Conv2D(64, (3,3), activation="relu", input_shape=input_shape),
            MaxPooling2D((2, 2)),
            Flatten(),
            Dense(128, activation='relu'),
            Dense(classes, activation="softmax")
    ])

    return model

def save_model(path, model):
    # Saves the model in a specified directory (path)
    model.save(path)


class myCallback(tf.keras.callbacks.Callback):
    
    # Stops the training at 99.9% accuracy
    def on_epoch_end(self, epoch, logs={}):
        if logs.get('accuracy') and (logs.get('accuracy')>0.999):
            print("\nReached 99% accuracy so stopping training!")
            self.model.stop_training = True