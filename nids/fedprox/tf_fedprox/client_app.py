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
"""Network intrusion detection system with FedProx strategy using Flower and TensorFlow.
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
from keras.losses import Loss, BinaryCrossentropy 
from tf_common.utility_functions import set_seed, load_data
from tf_common.ann_models import  load_model, compile_model

# Regularization term for FedProx. It computes the proximal term as the squared L2 norm between
# the local model weights and the global model weights, multiplied by mu/2.
class FedProxLoss(Loss):
    def __init__(self, 
                 local_model, 
                 global_weights, 
                 mu=0., 
                 reduction="sum_over_batch_size", 
                 name=None):
        super().__init__(name=name, reduction=reduction) 
        self.mu = mu
        self.actual_loss = BinaryCrossentropy()
        self.local_model = local_model 
        self.global_weights = global_weights

    # The tf.norm() function is unstable and generates tiny numerical error probably 
    # due to sqrt(). Let's use tf.reduce_sum() instead that in the end gives the same result.
    def __call__(self, Y_actual, Y_predicted, sample_weight=None):
        value = 0. 
        for w1, w2 in zip(self.local_model.trainable_weights, self.global_weights): 
            value += tf.reduce_sum(tf.square(tf.convert_to_tensor(w1) - tf.convert_to_tensor(w2))) 
            
        final_loss = (self.mu/2.) * value + self.actual_loss(Y_actual, Y_predicted) 

        return final_loss

    def get_config(self):
        return {'mu': float(self.mu)}


# Flower ClientApp
app = ClientApp()

@app.train()
def train(msg: Message, context: Context):
    # Configure logging
    update_console_handler(level=INFO, timestamps=True)

    log(INFO, "Received training message from FedProx server. Starting local training...")
    client = {}
    client["data_folder"] = str(context.node_config["data_folder"])
    client['name'] =  str(context.node_config['name'])
    
    client['rn_seed'] = int(context.run_config["rn_seed"])
    client["optimizer"] = str(context.run_config["optimizer"])
    client["epochs"] = int(context.run_config["epochs"])
    client["batch_size"] = int(context.run_config["batch_size"])

    client["proximal_mu"] = float(msg.content["config"]["proximal_mu"])
    client["server_round"] = int(msg.content["config"]["server_round"])

    log(INFO, f"=========  Starting round {client['server_round']} with configuration: ===========")
    log(INFO, f"Client name: {client['name']}")
    log(INFO, f"Optimizer={client['optimizer']}")
    log(INFO, f"Epochs={client['epochs']}")
    log(INFO, f"Batch_size={client['batch_size']}")
    log(INFO, f"rn_seed={client['rn_seed']}")
    log(INFO, f"Input_folder={client['data_folder']}")
    log(INFO, f"Proximal_mu={client['proximal_mu']}")

    # Set the seed for reproducibility
    set_seed(client["rn_seed"])

    # Load the data 
    load_data(client)

    # Load the model and set its weights to the ones received from the server
    model = load_model()
    model.set_weights(msg.content["arrays"].to_numpy_ndarrays())

    compile_model(model, 
                  client["optimizer"], 
                  loss=FedProxLoss(model, model.get_weights(), mu=client["proximal_mu"]))
    
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

    # Get training metrics
    train_loss = history.history["loss"][-1] 
    val_loss = history.history["val_loss"][-1]   

    # Measure local f1 
    X_val, Y_val = client['validation'] 
    Y_pred = np.squeeze(model.predict(X_val, batch_size=2048) > 0.5)
    local_f1 = f1_score(Y_val, Y_pred)     

    log(INFO, f"Finished local training. Train loss: {train_loss:.4f}, Val loss: {val_loss:.4f}")

    # Pack and send the model weights and metrics back as a message.
    model_record = ArrayRecord(model.get_weights())
    metrics = MetricRecord({"train_loss": train_loss, 
                            "val_loss": val_loss, 
                            "samples": client['training_samples'],
                            "local_f1": float(local_f1),
                            "local_training_time": local_training_time})        
    content = RecordDict({"arrays": model_record, "metrics": metrics})
    return Message(content=content, reply_to=msg)

@app.evaluate()
def evaluate(msg: Message, context: Context):
    # Configure logging
    update_console_handler(level=INFO, timestamps=True)

    log(INFO, "Received evaluation message from FedProx server. Starting local evaluation...")
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