"""
Can be memory intensive, it is possible to run the datasets just one at a time though
"""

import os
import numpy as np
import nibabel as nib
from utils import dump_json

# full_dataset_path = 'dataset/Outputs/cpac/nofilt_noglobal/reho'
asd_dataset_path = 'asd_dataset/Outputs/cpac/nofilt_noglobal/reho'
control_dataset_path = 'control_dataset/Outputs/cpac/nofilt_noglobal/reho'

asd_output_path = '/home/elliot/PycharmProjects/abide/processed_data_files/asd_raw'
control_output_path = '/home/elliot/PycharmProjects/abide/processed_data_files/control_raw'

# full_dataset_files = os.listdir(full_dataset_path)
asd_dataset_files = os.listdir(asd_dataset_path)
control_dataset_files = os.listdir(control_dataset_path)

"""
Dataset exploration results:
408 asd_fv examples
476 control_fv examples
all of shape (61, 73, 61)
Pixel range appears to be float in [0, 1]
Max for ASD: 0.619
Max for CTL: 0.588
"""

asd_dict = dict()
ctl_dict = dict()

for path in asd_dataset_files:
    img = nib.load(os.path.join(asd_dataset_path, path))
    img_arr = np.asarray(img.get_fdata())
    base_name = path.split('.')[0]
    asd_dict[base_name] = img_arr
    img.uncache()
dump_json(filepath=asd_output_path, filename='raw_img_asd.json', output=asd_dict)

for path in control_dataset_files:
    img = nib.load(os.path.join(control_dataset_path, path))
    img_arr = np.asarray(img.get_fdata())
    base_name = path.split('.')[0]
    ctl_dict[base_name] = img_arr
    img.uncache()
dump_json(filepath=control_output_path, filename='raw_img_ctl.json', output=ctl_dict)
