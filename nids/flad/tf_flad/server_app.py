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
"""Network intrusion detection system with Flad strategy using Flower and TensorFlow.
   Server app.
"""

from logging import INFO
import os
import time

from flwr.common import log
from flwr.common.logger import update_console_handler
from flwr.app import ArrayRecord, ConfigRecord, Context
from flwr.serverapp import Grid, ServerApp

from strategies import flad, utilities
from tf_common.ann_models import load_model
from tf_common.utility_functions import set_seed

# Create the ServerApp
app = ServerApp()

@app.main()
def main(grid: Grid, context: Context) -> None:
    # Configure logging
    update_console_handler(level=INFO, timestamps=True)

    # Load config (load all paramters for logging purposes)
    log(INFO, "Loading configuration for Flad strategy...")
    client_names = str(context.run_config["client_names"])
    min_epochs = int(context.run_config["min_epochs"])
    max_epochs = int(context.run_config["max_epochs"])
    min_steps = int(context.run_config["min_steps"])
    max_steps = int(context.run_config["max_steps"])
    rn_seed = int(context.run_config["rn_seed"])
    patience = int(context.run_config["patience"])
    output_folder = str(context.run_config["output_folder"])
    optimizer = str(context.run_config["optimizer"])

    # Create, if needed, the output folder for logs and models
    if os.path.isdir(output_folder) == False:
        os.mkdir(output_folder) 
    output_folder = output_folder + f"/federated_training_flad_{rn_seed}-" + time.strftime("%Y%m%d-%H%M%S") + "/"    
    if os.path.isdir(output_folder) == False:
        os.mkdir(output_folder)

    # Save config parameters in the summary.txt file
    with open(output_folder + "/summary.txt", "w") as summary_file:
        summary_file.write(f"client names: {client_names}\n")
        summary_file.write(f"min epochs: {min_epochs}\n")
        summary_file.write(f"max epochs: {max_epochs}\n")
        summary_file.write(f"min steps: {min_steps}\n")
        summary_file.write(f"max steps: {max_steps}\n")
        summary_file.write(f"random seed: {rn_seed}\n")
        summary_file.write(f"patience: {patience}\n")
        summary_file.write(f"optimizer: {optimizer}\n")

    # Set the seed for reproducibility
    set_seed(rn_seed)
    
    # Load initial model
    log(INFO, "Loading initial model...")
    model = load_model()
    arrays = ArrayRecord(model.get_weights())

    # Define and start Flad strategy
    strategy = flad.Flad(client_names)

    # Create a train_config record
    train_config = ConfigRecord({
        "min_epochs": min_epochs,
        "max_epochs": max_epochs,
        "min_steps": min_steps,
        "max_steps": max_steps,
        "patience": patience,
    })

    tstart = time.time()
    result = strategy.start(grid=grid, 
                            initial_arrays=arrays, 
                            train_config=train_config, 
                            )
    tend = time.time()
    duration = tend - tstart
    # Add duration to the summary.txt file
    with open(output_folder + "summary.txt", "a") as f:
        f.write(f"duration = {duration:.2f} seconds\n")    
    
    # Save the final model and the training history
    ndarrays = result.arrays.to_numpy_ndarrays()
    model.set_weights(ndarrays)
    utilities.save_results(model, result, output_folder, rn_seed)
