import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from utils import load_json, train_valid_split_indices

asd_data_path = '/home/elliot/PycharmProjects/abide/processed_data_files/asd_fv/feature_extraction_dict.json'
ctl_data_path = '/home/elliot/PycharmProjects/abide/processed_data_files/control_fv/feature_extraction_dict.json'


def load_dataset(asd_path, ctl_path, seed=7):
    x_train = list()
    x_valid = list()
    y_train = list()
    y_valid = list()

    asd_data = load_json(asd_path)
    asd_keys = list(asd_data.keys())
    train_indices, valid_indices = train_valid_split_indices(max_index=len(asd_keys), random_seed=seed)
    for idx in train_indices:
        x_train.append(asd_data[asd_keys[idx]])
        y_train.append(1.0)  # all positive cases
    for idx in valid_indices:
        x_valid.append(asd_data[asd_keys[idx]])
        y_valid.append(1.0)  # all positive cases
    asd_data.clear()

    # now the negative case loop
    ctl_data = load_json(ctl_path)
    ctl_keys = list(ctl_data.keys())
    for idx in train_indices:
        x_train.append(ctl_data[ctl_keys[idx]])
        y_train.append(0.0)  # all negative cases
    for idx in valid_indices:
        x_valid.append(ctl_data[ctl_keys[idx]])
        y_valid.append(0.0)  # all negative cases
    ctl_data.clear()

    return np.asarray(x_train), np.asarray(y_train), np.asarray(x_valid), np.asarray(y_valid)


if __name__ == '__main__':
    x_train, y_train, x_valid, y_valid = load_dataset(asd_data_path, ctl_data_path)

    simple_model = tf.keras.models.Sequential([
        tf.keras.layers.LSTM(256, input_shape=(None, 1280)), #128 units, 1280 input_dim
        tf.keras.layers.BatchNormalization(),  # fuck batchnorm tho
        tf.keras.layers.Dense(2, activation='softmax')] #output dim 2
    )

    simple_model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    history = simple_model.fit(x_train, y_train,
              validation_data=(x_valid, y_valid),
              batch_size=16,
              epochs=100)

    history = history.history
    # plot_loss
    plt.plot(history['loss'], label='loss')
    plt.plot(history['val_loss'], label='val_loss')
    plt.xlabel('epochs')
    plt.legend(loc='best')
    plt.grid()
    plt.savefig('lstm3d_losses.png')
    plt.clf()

    # plot accuracy
    plt.plot(history['accuracy'], label='accuracy')
    plt.plot(history['val_accuracy'], label='val_accuracy')
    plt.xlabel('epochs')
    plt.legend(loc='best')
    plt.grid()
    plt.savefig('lstm3d_accuracy.png')
    plt.clf()
