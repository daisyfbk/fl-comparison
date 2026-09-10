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
from tensorflow.keras.layers import Input, Dense, Flatten, Layer
from tensorflow.keras.layers import Dropout
from tensorflow.keras.models import Model, Sequential
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import EarlyStopping

K.set_image_data_format('channels_last')

# disable GPUs for test reproducibility
tf.config.set_visible_devices([], 'GPU')

MLP_UNITS = 32
N = 32
P = 10
BATCH_SIZE = 2000

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

#
# Specific for FedALA.
#
def clip(w): 
    return tf.clip_by_value(w, 0., 1.) 

class ALALayer(Layer): 
    def __init__(self, internal_layer, global_weights, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Copy internal layer 
        self.internal_layer = type(internal_layer).from_config(internal_layer.get_config()) 
        self.internal_layer.build_from_config(internal_layer.get_build_config())
        
        # Initialize ALA parameters 
        self.ww = [self.add_weight(shape=old.shape, initializer='ones', trainable=True, name='ww_'+old.name) for old in internal_layer.weights] 

        # local and global weights will be set later after compiling
        self.local_weights = internal_layer.get_weights() 
        self.global_weights = global_weights 
        
    def call_with_adapted_gradient(self, x):
        """
        Custom forward+backward behavior for the ALALayer
        """
        # global_weights - local_weights 
        update = [g-l for g,l in zip(self.global_weights, self.local_weights)]
        local = self.local_weights
        
        # Clipping. Note: reduce final performance 
        for ww in self.ww:
            ww.assign(clip(ww))
        
        @tf.custom_gradient
        def adapted_gradient(x):
            # Compute gradient of layer weights wrt to ala parameters 
            with tf.GradientTape() as tape_w:
                # (global_weights - local_weights) * ww
                adapted_update = [ww * u for u, ww in zip(update, self.ww)]

                # local_weights + adapted_update
                new_w = [u + l for l, u in zip(local, adapted_update)] 
 
                # Assign new weights using the updated `new_w` list
                for i, new_weight in enumerate(new_w):
                    self.internal_layer.weights[i].assign(new_weight)

            # Compute output gradient wrt to layer weights 
            w = self.internal_layer.trainable_variables 
            with tf.GradientTape() as tape_y:
                tape_y.watch(x) 
                y = self.internal_layer(x)

            # The gradient should go to 'ww' weights, its computation use chain rule. 
            # The gradient of the original/internal layer are set to zero (no training) 
            def backward(dafter, variables):
                dy_dx, dy_dw = tape_y.gradient(y, [x, w], output_gradients=dafter) 
                dy_dww = tape_w.gradient(new_w, self.ww, output_gradients=dy_dw)

                grad_vars = []
                for var in variables:
                    if 'ww' in var.name:
                        if "bias" in var.name:
                            grad_vars.append(dy_dww[1]) 
                        else:
                            grad_vars.append(dy_dww[0])
                    else:
                        grad_vars.append(tf.zeros(var.shape))

                return dy_dx, grad_vars
            return y, backward

        return adapted_gradient(x) 

    def call(self, inputs, compile=False):
        return self.call_with_adapted_gradient(inputs) 

    def set_weights(self, local_weights, global_weights):
        self.local_weights = local_weights 
        self.global_weights = global_weights

def create_ala_model(local_model, global_model, ala_trainable_layers=0):
    assert len(local_model.layers) == len(global_model.layers)
    assert ala_trainable_layers >= 0 

    k = -ala_trainable_layers 
    ala_model = Sequential(name="AdaptedLocalAggregation")
    # Freeze lower layers 
    for local_layer in local_model.layers[:k]:
        # Clone to avoid sharing trainable state with local_model layers.
        cloned_layer = type(local_layer).from_config(local_layer.get_config())
        cloned_layer.trainable = False
        ala_model.add(cloned_layer)

    # Custom higher layers 
    for local_layer, global_layer in zip(local_model.layers[k:], global_model.layers[k:]): 
        if not isinstance(local_layer, Flatten):
            ala_model.add(ALALayer(local_layer, global_layer.get_weights()))
        else:
            ala_model.add(Flatten())
    
    return ala_model

def compile_ala_model(model: Model) -> None:
    optimizer = SGD(learning_rate=1.0, momentum=0.0, nesterov=False)
    model.compile(loss='binary_crossentropy', optimizer=optimizer,metrics=['accuracy'])


def extract_ala_weights(ala_model):
    ala_weights = []
    for layer in ala_model.layers: 
        if isinstance(layer, ALALayer): 
            for ws in layer.ww:
                ala_weights.append(ws) # Convert tf.Variable to Tensor 
        else:
            layer.trainable = True 
    # Clipping. Note: reduce final performance 
    ala_weights = [clip(ala) for ala in ala_weights]
    return ala_weights

# l + a*(g-l)
def reconstruct_weights(local_weights, global_weights, ala_weights):
    tmp1 = [g-l for l, g in zip(local_weights, global_weights)]
    # Note: ala_weights may be shorter than local and global ones => we keep only higher layers weights
    tmp2 = [a*g for a, g in zip(ala_weights, tmp1[-len(ala_weights):])] 
    combined_weights =  [l + u for l, u in zip(local_weights[-len(ala_weights):], tmp2)] 
    return combined_weights     

def initialize_ala_model(local_model,
                         global_model, 
                         ala_model,
                         training_data,
                         ala_sample_ratio,
                         ala_trained): 
    local_weights = local_model.get_weights() 
    global_weights = global_model.get_weights() 

    # Update local and global weights 
    for layer, local_layer, global_layer in zip(ala_model.layers, local_model.layers, global_model.layers):
        if isinstance(layer, ALALayer):
            layer.set_weights(local_layer.get_weights(), global_layer.get_weights()) 

    # Compile ala_model 
    ala_model.compile(loss='binary_crossentropy', 
                      optimizer=SGD(learning_rate=1.0, momentum=0.0, nesterov=False), 
                      metrics=['accuracy'])

    # Sample s% of random train data 
    boolean_mask = np.full(len(training_data[0]), False)
    boolean_mask[:int(len(training_data[1]) * ala_sample_ratio)] = True
    np.random.shuffle(boolean_mask) 
    random_x = tf.boolean_mask(training_data[0], boolean_mask) 
    random_y = tf.boolean_mask(training_data[1], boolean_mask) 

    if not ala_trained:
        # Until convergence 
        overfitCallback = EarlyStopping(monitor='loss', min_delta=0.1)
        history = ala_model.fit(x=random_x, 
                                y=random_y, 
                                epochs=100, 
                                batch_size=BATCH_SIZE, 
                                verbose=2, 
                                callbacks=[overfitCallback])  
    else: 
        history = ala_model.fit(x=random_x, 
                                y=random_y, 
                                epochs=1, 
                                batch_size=BATCH_SIZE, 
                                verbose=2, 
                                callbacks=[]) 

    # Apply learned ala_weights to local model 
    ala_weights = extract_ala_weights(ala_model) 

    # Reverse list of weights 
    new_weights = reconstruct_weights(local_weights, global_weights, ala_weights)[::-1] 
    i = 0 
    for client_layer in reversed(local_model.layers):
        # Each layer may have multiple weights 
        new_weight = new_weights[i:i+len(client_layer.weights)][::-1] # Reverse back
        client_layer.set_weights(new_weight)
        i += len(client_layer.weights)
        # Exit if new_weights is finished
        if i >= len(new_weights):
            break 