#
# Copyright 2026 Fondazione Bruno Kessler.
#  
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#  
#      http://www.apache.org/licenses/LICENSE-2.0
#  
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#  
#


import os

from  logging import INFO
import glob
import h5py

import random
import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle

from flwr.common import log

TIME_WINDOW = 10
MAX_FLOW_LEN = 10
CLASSES = np.array([0, 1])


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility."""

    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def load_data(client: dict, only_validation: bool = False) -> None:
    log(INFO, f"Loading data for client from folder {client['data_folder']} with seed {client['rn_seed']}.")
    if not only_validation:
        X_train, Y_train = load_set(client['data_folder'], 
                                    "train", 
                                    client['rn_seed'])
        X_train_tensor = tf.convert_to_tensor(X_train, dtype=tf.float32)
        client['training'] = (X_train_tensor, Y_train)
        client['training_samples'] = client['training'][1].shape[0]
    
    X_val, Y_val = load_set(client['data_folder'], 
                            "val", 
                            client['rn_seed'])
    X_val_tensor = tf.convert_to_tensor(X_val, dtype=tf.float32)
    client['validation'] = (X_val_tensor, Y_val)
    client['validation_samples'] = client['validation'][1].shape[0]

def load_dataset(filename: str) -> tuple[np.ndarray, np.ndarray]:
    dataset = h5py.File(filename, "r")
    set_x_orig = np.array(dataset['set_x'][:])  # features
    set_y_orig = np.array(dataset['set_y'][:])  # labels

    if len(set_x_orig.shape) == 3: # array-like data with no channels
        X_train = np.reshape(set_x_orig, (set_x_orig.shape[0], set_x_orig.shape[1], set_x_orig.shape[2], 1))
    else:
        X_train = set_x_orig
    Y_train = set_y_orig

    return X_train, Y_train

def load_set(data_folder: str, 
             set_type: str, 
             seed: int) -> tuple[np.ndarray, np.ndarray]:
    set_list = []

    # Assume only one folder where data files are stored.
    files = glob.glob(data_folder + "/*" + '-' + set_type + '.hdf5')
    for file in files:
        filename = file.split('/')[-1].strip()
        tw = int(filename.split('-')[0].strip().replace('t', ''))
        mfl = int(filename.split('-')[1].strip().replace('n', ''))
        dn = filename.split('-')[2].strip()
        if tw != TIME_WINDOW:
            raise ValueError("Mismatching time window size among datasets!")
        if mfl != MAX_FLOW_LEN:
            raise ValueError("Mismatching flow length size among datasets!")

        X, Y = load_dataset(file)
        if not np.array_equal(np.unique(Y), CLASSES):
            raise ValueError("Mismatching classes among datasets!")
        set_list.append((X, Y))

    # Concatenation of all the training and validation sets
    X = set_list[0][0]
    Y = set_list[0][1]
    for n in range(1, len(set_list)):
        X = np.concatenate((X, set_list[n][0]), axis=0)
        Y = np.concatenate((Y, set_list[n][1]), axis=0)

    X, Y = shuffle(X, Y, random_state = seed)
    return X, Y
