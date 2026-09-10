#
# Copyright 2025 Fondazione Bruno Kessler.
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

from typing import Optional
from logging import INFO
import numpy as np

from flwr.common import log

import tensorflow as tf
config = tf.compat.v1.ConfigProto(inter_op_parallelism_threads=1)
from tensorflow.keras.optimizers import Adam,SGD
from tensorflow.keras.layers import Input, Dense, Activation,  Flatten, Conv2D
from tensorflow.keras.layers import MaxPooling2D, Dropout
from tensorflow.keras.models import Model, Sequential
import tensorflow.keras.backend as K

K.set_image_data_format('channels_last')

# disable GPUs for test reproducibility
tf.config.set_visible_devices([], 'GPU')

MLP_UNITS = 32
N = 32
P = 10

# MPL model
def FCModel(model_name: str, 
            input_shape: tf.TensorShape, 
            units: int, 
            classes: int = 1, 
            dropout: Optional[float] = None) -> Model:
    K.clear_session()

    model = Sequential(name=model_name)

    model.add(Input(shape=input_shape))
    model.add(Flatten())
    model.add(Dense(units, activation='relu', name='fc0'))
    if dropout is not None and isinstance(dropout, float):
        model.add(Dropout(dropout))
    model.add(Dense(units, activation='relu', name='fc1'))
    if dropout is not None and isinstance(dropout, float):
        model.add(Dropout(dropout))
    model.add(Dense(classes, activation='sigmoid', name='fc3'))

    log(INFO, model.summary())
    return model

def compile_model(model: Model, 
                  optimizer_type: str ="SGD",
                  loss: str ='binary_crossentropy') -> None:
    if optimizer_type == "Adam":
        optimizer = Adam(learning_rate=0.01, beta_1=0.9, beta_2=0.999)
    else:
        optimizer = SGD(learning_rate=0.1, momentum=0.0, nesterov=False)

    model.compile(loss=loss, optimizer=optimizer,metrics=['accuracy'])

def load_model()-> Model: 
    # Input shape is fixed and predefined by the server.
    input_shape = tf.TensorShape([P, N, 1])
    log(INFO, f"Loading model with input shape {input_shape}...")
    model = FCModel('mlp', input_shape, units=MLP_UNITS)

    return model