import os
import json
import jsonpickle
import numpy as np

def dump_json(filepath, filename, output):
    os.makedirs(filepath, exist_ok=True)
    with open(os.path.join(filepath, filename), 'w') as out:
        json.dump(jsonpickle.encode(output), out, indent=4)
    out.close()


def load_json(filepath):
    with open(filepath) as file:
        js_string = json.load(file)
    file.close()
    return jsonpickle.decode(js_string)


def train_valid_split_indices(max_index, validation_percentage=0.3, min_index=0, random_seed=None):
    """
    Given a range of data examples, randomly assigns indices to being either for training or validation
    :param max_index: max index to be assigned (usually length of the data array)
    :param validation_percentage: probability that any individual index will be assigned to the validation set
    :param min_index: starting index, defaults to 0
    :return: list of train indices, list of validation indices
    """
    assert(max_index - min_index) > 1, 'Too few examples'
    print('Using random seed ' + str(random_seed) + ' for train/valid split')
    np.random.seed(seed=random_seed)
    train_indices = list()
    valid_indices = list()
    for i in range(min_index, max_index):
        if np.random.uniform() < validation_percentage:
            valid_indices.append(i)
        else:
            train_indices.append(i)
    # sanity check that both lists have at least one index, prevents error with small test datasets
    if len(train_indices) == 0:
        train_indices.append(valid_indices.pop())
    elif len(valid_indices) == 0:
        valid_indices.append(train_indices.pop())
    return train_indices, valid_indices


recovered = load_json('/home/elliot/PycharmProjects/abide/processed_data_files/asd_raw/raw_img_asd.json')