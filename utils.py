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
# recovered = load_json('/home/elliot/PycharmProjects/abide/processed_data_files/asd_raw/raw_img_asd.json')