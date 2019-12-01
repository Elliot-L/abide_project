"""
Some inspiration taken from this repo:
https://github.com/tensorflow/docs/blob/master/site/en/tutorials/images/transfer_learning.ipynb

Transforms the 3D images into a set of feature vectors extracted via MobileNet v2
"""


import os
from skimage.transform import resize
import numpy as np
import nibabel as nib
import tensorflow as tf
keras = tf.keras
from utils import dump_json

asd_dataset_path = '/home/elliot/PycharmProjects/abide/asd_dataset/Outputs/cpac/nofilt_noglobal/reho'
control_dataset_path = '/home/elliot/PycharmProjects/abide/control_dataset/Outputs/cpac/nofilt_noglobal/reho'

asd_output_path = '/home/elliot/PycharmProjects/abide/processed_data_files/asd_fv'
control_output_path = '/home/elliot/PycharmProjects/abide/processed_data_files/control_fv'

IMG_SIZE = 96  # All images will be resized to IMG_SIZExIMG_SIZE
IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)
num_slices = 61 # hard-coded for the dataset


def prep_image_batch(img_slices):
    """
    :param img_slices: (61, 73, 61) 3D brain image
    :return: a batch of processed 2D images (one per slice) for Mobilenet feature extraction
    """
    img_batch = list()
    # todo: remap using global max ~0.6 -> 1.0?
    # re-map pixel values from [0, 1] to [-1, 1]
    img_slices = (img_slices * 2.0) - 1.0
    for i in range(num_slices):
        img = img_slices[:, :, i]
        # resize image, broadcast into 3 channels
        img = np.repeat(np.expand_dims(resize(img, (IMG_SIZE, IMG_SIZE)), -1), 3, axis=-1)
        img_batch.append(img)
    return np.asarray(img_batch) # now has batch_dim = num slices, IMG_SIZE rows and cols, and 3 identical channels


if __name__ == '__main__':
    # loading an example image to get the flow right
    asd_dataset_files = os.listdir(asd_dataset_path)
    ctl_dataset_files = os.listdir(control_dataset_path)

    # Create the base model from the pre-trained model MobileNet V2
    base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                                   include_top=False,
                                                   weights='imagenet')
    # last pooling layer to get down to a 1D feature vector
    global_average_layer = tf.keras.layers.GlobalAveragePooling2D()

    asd_fv_dict = dict()
    for img_file in asd_dataset_files:
        image = np.asarray(nib.load(os.path.join(asd_dataset_path, img_file)).get_fdata())
        img_batch = prep_image_batch(image)
        fvs = global_average_layer(base_model(img_batch))
        base_name = img_file.split('.')[0]
        asd_fv_dict[base_name] = fvs.numpy()
    dump_json(filepath=asd_output_path, filename='feature_extraction_dict.json', output=asd_fv_dict)

    ctl_fv_dict = dict()
    for img_file in ctl_dataset_files:
        image = np.asarray(nib.load(os.path.join(control_dataset_path, img_file)).get_fdata())
        img_batch = prep_image_batch(image)
        fvs = global_average_layer(base_model(img_batch))
        base_name = img_file.split('.')[0]
        ctl_fv_dict[base_name] = fvs.numpy()
    dump_json(filepath=control_output_path, filename='feature_extraction_dict.json', output=ctl_fv_dict)

    print('Done!')