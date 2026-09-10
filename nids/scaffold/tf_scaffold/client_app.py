# 
# Derivative work:
# Copyright 2026 Fondazione Bruno Kessler
# 
# Original/Previous work:
# Copyright 2025 Flower Labs GmbH.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#  
#     http://www.apache.org/licenses/LICENSE-2.0
#  
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#  
"""Network intrusion detection system with Scaffold strategy using Flower and TensorFlow.
   Client app.
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import time
from logging import INFO
import numpy as np
import tensorflow as tf

from flwr.common import log
from flwr.common.logger import update_console_handler
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict, ConfigRecord
from flwr.clientapp import ClientApp
from sklearn.metrics import f1_score 
from keras.optimizers import SGD
from tf_common.utility_functions import set_seed, load_data
from tf_common.ann_models import  load_model, compile_model


class SCAFFOLD_SGD(SGD):
    def __init__(self, local_variate, global_variate, **kwargs):
        super().__init__(**kwargs)
        self.correction = [g - l for g, l in zip(global_variate, local_variate)] 

    def update_step(self, gradient, variable, learning_rate):
        super().update_step(gradient, variable, learning_rate)
        # Correct variable with variate 
        correction = self.correction[self._get_variable_index(variable)] 
        self.assign_add(variable, correction * learning_rate) 

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "correction": self.correction, 
            }
        )
        return config


# Flower ClientApp
app = ClientApp()

@app.train()
def train(msg: Message, context: Context):
    # Configure logging
    update_console_handler(level=INFO, timestamps=True)

    log(INFO, "Received training message from Scaffold server. Starting local training...")
    client = {}
    client['name'] =  str(context.node_config['name'])
    client["data_folder"] = str(context.node_config["data_folder"])
    
    client['rn_seed'] = int(context.run_config["rn_seed"])
    client["optimizer"] = str(context.run_config["optimizer"])
    client["epochs"] = int(context.run_config["epochs"])]
    client["batch_size"] = int(context.run_config["batch_size"])

    client["server_round"] = int(msg.content["config"]["server_round"])

    log(INFO, f"==========  Starting round {client['server_round']} with configuration: ============")
    log(INFO, f"Client name: {client['name']}")
    log(INFO, f"Optimizer={client['optimizer']}")
    log(INFO, f"Epochs={client['epochs']}")
    log(INFO, f"Batch_size={client['batch_size']}")
    log(INFO, f"rn_seed={client['rn_seed']}")
    log(INFO, f"Input_folder={client['data_folder']}")

    # Set the seed for reproducibility
    set_seed(client["rn_seed"])

    # Load the data 
    load_data(client)

    # Load the model and set its weights to the ones received from the server
    model = load_model()
    model.set_weights(msg.content["arrays"].to_numpy_ndarrays())

    if context.state.get("local_variate") is not None:
        client['local_variate'] = context.state["local_variate"].to_numpy_ndarrays()
    else:
        client['local_variate'] = [np.zeros(w.shape) for w in msg.content["arrays"].to_numpy_ndarrays()]
    
    client['global_variate'] = msg.content["variate_arrays"].to_numpy_ndarrays()


    custom_optimizer = SCAFFOLD_SGD(client['local_variate'], 
                                    client['global_variate'], 
                                    learning_rate=0.1, 
                                    momentum=0.0, 
                                    nesterov=False) 

    compile_model(model, custom_optimizer, 'binary_crossentropy')
    
    # Train the model
    tstart = time.time()
    history = model.fit(
        x = client['training'][0],
        y = client['training'][1],
        validation_data = (client['validation'][0], client['validation'][1]), 
        epochs = client["epochs"],
        batch_size = client["batch_size"],
        verbose = 2, 
        callbacks = [],
    )

    tend = time.time()
    local_training_time = tend - tstart

    watched_x = client['validation'][0]
    with tf.GradientTape(persistent=False) as tape:
        tape.watch(watched_x) 
        watched_y = model(watched_x) 
    
    new_variate = tape.gradient(watched_y, [w for w in model.weights])
    new_variate[:] = [new/len(watched_x) for new in new_variate]
    client['delta_variate'] = [new - old for new, old in zip(new_variate, client['local_variate'])] 
    client['local_variate'][:] = new_variate 
    local_variate_arrays = [v.numpy() if tf.is_tensor(v) else np.asarray(v) for v in client['local_variate']]
    context.state["local_variate"] = ArrayRecord(local_variate_arrays)
    del tape 

    # Get training metrics
    train_loss = history.history["loss"][-1] 
    val_loss = history.history["val_loss"][-1]

    # Measure local f1 
    X_val, Y_val = client['validation'] 
    Y_pred = np.squeeze(model.predict(X_val, batch_size=2048) > 0.5)
    local_f1 = f1_score(Y_val, Y_pred) 

    log(INFO, f"Finished local training. Train loss: {train_loss:.4f}, Val loss: {val_loss:.4f}")

    # Pack and send the model weights, the delta_variate and metrics back as a message.
    delta_variate_arrays = [v.numpy() if tf.is_tensor(v) else np.asarray(v) for v in client['delta_variate']]
    variate_record = ArrayRecord(delta_variate_arrays)
    model_record = ArrayRecord(model.get_weights())
    metrics = MetricRecord({"train_loss": train_loss, 
                            "val_loss": val_loss, 
                            "samples": len(client['training'][1]),
                            "local_f1": float(local_f1),
                            "local_training_time": local_training_time})    
    content = RecordDict({"arrays": model_record, 
                          "variate_arrays": variate_record, 
                          "metrics": metrics})
    return Message(content=content, reply_to=msg)

@app.evaluate()
def evaluate(msg: Message, context: Context):
    # Configure logging
    update_console_handler(level=INFO, timestamps=True)

    log(INFO, "Received evaluation message from Scaffold server. Starting local evaluation...")
    client = {}
    client["name"] = str(context.node_config["name"])
    client["data_folder"] = str(context.node_config["data_folder"])
    client['rn_seed'] = int(context.run_config["rn_seed"])

    # Set the seed for reproducibility
    set_seed(client["rn_seed"])

    # Load the data 
    load_data(client, only_validation=True)

    # Load the model and set its weights to the ones received from the server
    model = load_model()
    model.set_weights(msg.content["arrays"].to_numpy_ndarrays())

    # Evaluate the model
    X_val, Y_val = client['validation']
    Y_pred = np.squeeze(model.predict(X_val, batch_size=2048) > 0.5)
    client_f1 = f1_score(Y_val, Y_pred)

    # Pack and send the model weights and metrics as a message
    metrics = MetricRecord({"f1_score": float(client_f1)})
    content = RecordDict({"metrics": metrics})
    return Message(content=content, reply_to=msg)

@app.query("info")
def info(msg: Message, context: Context) -> Message:
    # Return the client name
    client_name = context.node_config['name']
    log(INFO, f"Received info query from FedAvg server. Responding with client name: {client_name}")
    content = RecordDict({
        "config": ConfigRecord({"name": client_name}),
    })

    return Message(content=content, reply_to=msg)