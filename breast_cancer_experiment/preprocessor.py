import os
import numpy as np
import imageio

"""
Dataset exploration results:
Images are 8bit integers in [0, 228] -> we will remap to floats in [-1, 1]

Nested dict: first one has sample names as keys pointing to dicts w/ individual sample info
Each nested dict has the actual image, as well as label info 
"""


def preprocess_mammograms(triplicate_channels=False, label_property='normal'):
    assert label_property in ['normal', 'benign']
    image_info_dict = dict()

    with open('data_stuff.txt', 'r') as label_lines:
        line = label_lines.readline()
        while line:
            sample_name = line.split()[0]
            normal = line.split()[2] == 'NORM'
            benign = True
            if len(line.split()) > 3:
                benign = line.split()[3] == 'B'
            image_info_dict[sample_name] = {'normal': normal, 'benign': benign}
            line = label_lines.readline()

    images = list()
    for image_name in os.listdir('images'):
        if 'pgm' in image_name:
            image = imageio.imread(os.path.join('images', image_name))
            """
            Remapping image to be in [-1, 1]
            Optional triplicating of channels for transfer learning 
            """
            image = (np.asarray(image) / 228.0) * 2.0 - 1.0
            if triplicate_channels:
                image = np.repeat(np.expand_dims(image, axis=-1), 3, axis=-1)
            images.append((image_name.split('.')[0], image))


# now split the images and label according to the desired property


print('done')