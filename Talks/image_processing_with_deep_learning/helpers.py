import keras
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import RMSprop
from matplotlib import gridspec


def show_weights(layer):
    nrow = 2
    ncol = 5
    w = layer.get_weights()[0]
    plt.figure(figsize=(10, 4))
    gs = gridspec.GridSpec(nrow, ncol, wspace=0.1, hspace=0.1)
    for i in range(nrow):
        for j in range(ncol):
            im = w[:, i * 5 + j]
            ax = plt.subplot(gs[i, j])
            ax.imshow(im.reshape(28, 28), cmap='gray')
            ax.set_xticks([])
            ax.set_yticks([])
    plt.show()


def get_mnist_demo():
    # the data, shuffled and split between train and test sets
    (x_train, y_train), _ = mnist.load_data()

    x_train = x_train.reshape(60000, 784)
    x_train = x_train.astype('float32')
    x_train /= 255

    x_train = x_train[:1000]
    y_train = y_train[:1000]

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, 10)

    model = Sequential()
    model.add(Dense(10, activation='softmax', input_shape=(784,)))
    model.compile(loss='categorical_crossentropy', optimizer=RMSprop(), metrics=['accuracy'])
    return model, x_train, y_train
