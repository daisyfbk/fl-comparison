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
"""Network intrusion detection system with FedSBS strategy using Flower and TensorFlow.
   Server app.
"""

from logging import INFO
import os
import time

from flwr.common import log
from flwr.common.logger import update_console_handler
from flwr.app import ArrayRecord, Context
from flwr.serverapp import Grid, ServerApp

from strategies import fedsbs, utilities
from tf_common.utility_functions import set_seed
from tf_common.ann_models import load_model

# Create the ServerApp
app = ServerApp()

@app.main()
def main(grid: Grid, context: Context) -> None:
    # Configure logging
    update_console_handler(level=INFO, timestamps=True)

    # Load config (load all paramters for logging purposes)
    log(INFO, "Loading configuration for FedSBS strategy...")
    client_names = str(context.run_config["client_names"])
    num_rounds = context.run_config["rounds"]
    fraction_train = context.run_config["fraction-train"]
    epsilon = context.run_config["epsilon"]
    epsilon_min = context.run_config["epsilon_min"]
    temperature = context.run_config["temperature"]
    server_learning_rate = context.run_config["server_learning_rate"]
    beta_momentum = context.run_config["beta_momentum"]
    rn_seed = int(context.run_config["rn_seed"])
    proximal_mu = float(context.run_config["proximal_mu"])
    output_folder = str(context.run_config["output_folder"])
    epochs = int(context.run_config["epochs"])
    batch_size = int(context.run_config["batch_size"])
    optimizer = str(context.run_config["optimizer"])    

    # Create, if needed, the output folder for logs and models
    if os.path.isdir(output_folder) == False:
        os.mkdir(output_folder) 
    output_folder = output_folder + f"/federated_training_fedsbs_{rn_seed}-" + time.strftime("%Y%m%d-%H%M%S") + "/"    
    if os.path.isdir(output_folder) == False:
        os.mkdir(output_folder)
    # Save config parameters in the summary.txt file
    with open(output_folder + "summary.txt", "w") as f:
        f.write(f"client_names = {client_names}\n")
        f.write(f"rounds = {num_rounds}\n")
        f.write(f"fraction_train = {fraction_train}\n")
        f.write(f"epsilon = {epsilon}\n")
        f.write(f"epsilon_min = {epsilon_min}\n")
        f.write(f"temperature = {temperature}\n")
        f.write(f"server_learning_rate = {server_learning_rate}\n")
        f.write(f"beta_momentum = {beta_momentum}\n")
        f.write(f"rn_seed = {rn_seed}\n")
        f.write(f"proximal_mu = {proximal_mu}\n")
        f.write(f"epochs = {epochs}\n")
        f.write(f"batch_size = {batch_size}\n")
        f.write(f"optimizer = {optimizer}\n")
    
    # Set the seed for reproducibility
    set_seed(rn_seed)
    
    # Load initial model
    log(INFO, "Loading initial model...")
    model = load_model()
    arrays = ArrayRecord(model.get_weights())

    # Define and start FedSBS strategy
    strategy = fedsbs.FedSBS(client_names=client_names,
                             rounds=num_rounds,
                             fraction_train=fraction_train, 
                             rn_seed=rn_seed,
                             proximal_mu=proximal_mu,
                             epsilon=epsilon,
                             epsilon_min=epsilon_min,
                             temperature=temperature,
                             server_learning_rate=server_learning_rate,
                             beta_momentum=beta_momentum)

    tstart = time.time()
    result = strategy.start(grid=grid, 
                            initial_arrays=arrays, 
                            num_rounds=num_rounds, 
                            )
    
    tend = time.time()
    duration = tend - tstart
    # Add duration to the summary.txt file
    with open(output_folder + "summary.txt", "a") as f:
        f.write(f"duration = {duration:.2f} seconds\n")

    # Save the final/last model (not the best one) and the training history.
    ndarrays = result.arrays.to_numpy_ndarrays()
    model.set_weights(ndarrays)
    utilities.save_results(model, result, output_folder, rn_seed)

