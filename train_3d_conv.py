import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from utils import load_json, load_dataset

asd_data_path = '/home/elliot/PycharmProjects/abide/processed_data_files/asd_raw/debug_raw_img_asd.json'
ctl_data_path = '/home/elliot/PycharmProjects/abide/processed_data_files/control_raw/debug_raw_img_ctl.json'

from tensorflow.keras.optimizers import Adam

lr = 0.0001

if __name__ == '__main__':
    x_train, y_train, x_valid, y_valid = load_dataset(asd_data_path, ctl_data_path)
    assert tf.keras.backend.image_data_format() == 'channels_last'
    x_train = np.expand_dims(x_train, axis=-1)
    x_valid = np.expand_dims(x_valid, axis=-1)
    input_shape = (61, 73, 61, 1)
    model = tf.keras.models.Sequential()
    model.add(layers.Conv3D(32, kernel_size=(3, 3, 3), activation='relu', input_shape=input_shape))
    model.add(layers.Conv3D(32, kernel_size=(3, 3, 3), activation='relu'))
    model.add(layers.MaxPooling3D(pool_size=(2, 2, 2)))

    # model.add(layers.Dropout(0.25))
    # model.add(layers.Conv3D(64, kernel_size=(3, 3, 3), activation='relu'))
    # model.add(layers.Conv3D(64, kernel_size=(3, 3, 3), activation='relu'))
    # model.add(layers.MaxPooling3D(pool_size=(2, 2, 2)))
    model.add(layers.Dropout(0.25))
    model.add(layers.Flatten())
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dense(2, activation='softmax'))

    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=Adam(lr=lr),
                  metrics=['accuracy'])

    history = model.fit(x_train, y_train,
              validation_data=(x_valid, y_valid),
              batch_size=1,
              epochs=100)

    history = history.history
    # plot_loss
    plt.plot(history['loss'], label='loss')
    plt.plot(history['val_loss'], label='val_loss')
    plt.xlabel('epochs')
    plt.legend(loc='best')
    plt.grid()
    plt.savefig('conv3d_losses.png')
    plt.clf()

    # plot accuracy
    plt.plot(history['accuracy'], label='accuracy')
    plt.plot(history['val_accuracy'], label='val_accuracy')
    plt.xlabel('epochs')
    plt.legend(loc='best')
    plt.grid()
    plt.savefig('conv3d_accuracy.png')
    plt.clf()
