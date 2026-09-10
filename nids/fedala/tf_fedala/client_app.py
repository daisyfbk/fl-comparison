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
"""Network intrusion detection system with FedALA strategy using Flower and TensorFlow.
   Client app.
"""

import os
import time
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from logging import INFO
import numpy as np

from flwr.common import log
from flwr.common.logger import update_console_handler
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict, ConfigRecord
from flwr.clientapp import ClientApp
from sklearn.metrics import f1_score 
from tf_common.utility_functions import set_seed, load_data
from tf_fedala.ann_models import load_model, create_ala_model, compile_ala_model, compile_model, initialize_ala_model

# Flower ClientApp
app = ClientApp()

@app.train()
def train(msg: Message, context: Context):
    # Configure logging
    update_console_handler(level=INFO, timestamps=True)

    log(INFO, "Received training message from FedALA server. Starting local training...")
    client = {}
    client['name'] =  str(context.node_config['name'])
    client["data_folder"] = str(context.node_config["data_folder"])
    
    client['rn_seed'] = int(context.run_config["rn_seed"])
    client["optimizer"] = str(context.run_config["optimizer"])
    client["epochs"] = int(context.run_config["epochs"])
    client["batch_size"] = int(context.run_config["batch_size"])

    client["server_round"] = int(msg.content["config"]["server_round"])
    client["ala_trainable_layers"] = int(msg.content["config"]["ala_trainable_layers"])
    client["ala_sample_ratio"] = float(msg.content["config"]["ala_sample_ratio"])

    log(INFO, f"========  Starting round {client['server_round']} with configuration: =========")
    log(INFO, f"Client name: {client['name']}")
    log(INFO, f"Optimizer={client['optimizer']}")
    log(INFO, f"Epochs={client['epochs']}")
    log(INFO, f"Batch_size={client['batch_size']}")
    log(INFO, f"rn_seed={client['rn_seed']}")
    log(INFO, f"Input_folder={client['data_folder']}")
    log(INFO, f"ala_trainable_layers={client['ala_trainable_layers']}")
    log(INFO, f"ala_sample_ratio={client['ala_sample_ratio']}")

    # Set the seed for reproducibility
    set_seed(client["rn_seed"])

    # Load the data 
    load_data(client)

    # Load the model and set its weights to the ones received from the server
    client['global_model'] = load_model()
    client['global_model'].set_weights(msg.content["arrays"].to_numpy_ndarrays())
 
    if context.state.get("model_weights") is not None:
        client['local_model'] = load_model()
        client['local_model'].set_weights(context.state["model_weights"].to_numpy_ndarrays())
        client['ala_model'] = create_ala_model(client['local_model'], client['global_model'], client["ala_trainable_layers"])
        compile_ala_model(client['ala_model'])  
        model_to_train = client['local_model']
    else:  
        model_to_train = client['global_model']  

    # Training time must comprise also the initialization of the ALA model.
    tstart = time.time()
    if context.state.get("model_weights") is not None:
        client['global_model'].trainable = False
        client['local_model'].trainable = False
        if client["server_round"] == 2:  # Full training only during second round.
            initialize_ala_model(client['local_model'],
                                client['global_model'], 
                                client['ala_model'],
                                client['training'],
                                client['ala_sample_ratio'],
                                ala_trained=False)
        else:
            initialize_ala_model(client['local_model'],
                                client['global_model'], 
                                client['ala_model'],
                                client['training'],
                                client['ala_sample_ratio'],
                                ala_trained=True)  

    model_to_train.trainable = True
    compile_model(model_to_train, client['optimizer'], 'binary_crossentropy')
        
    # Train the model
    history = model_to_train.fit(
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

    # Save current local model in the context for the next round.
    context.state["model_weights"] = ArrayRecord(model_to_train.get_weights())

    # Get training metrics
    train_loss = history.history["loss"][-1] 
    val_loss = history.history["val_loss"][-1]

    # Measure local f1 
    X_val, Y_val = client['validation'] 
    Y_pred = np.squeeze(model_to_train.predict(X_val, batch_size=2048) > 0.5)
    local_f1 = f1_score(Y_val, Y_pred) 

    log(INFO, f"Finished local training. Train loss: {train_loss:.4f}, Val loss: {val_loss:.4f}")

    # Pack and send the model weights and metrics back as a message.
    model_record = ArrayRecord(model_to_train.get_weights())
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

    log(INFO, "Received evaluation message from FedALA server. Starting local evaluation...")
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

