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

import pickle
from logging import INFO
import csv

from tensorflow.keras.models import Model

from flwr.serverapp.strategy import Result
from flwr.common import (
    ArrayRecord,
    ConfigRecord,
    MetricRecord,
    log,
)


def reset_round_client_metrics(client: dict):
    """Reset the message size and local metrics for a client."""
    client['train_sent'] = 0
    client['evaluate_sent'] = 0
    client['train_received'] = 0
    client['evaluate_received'] = 0
    client['local_f1'] = -1
    client['local_training_time'] = -1

def compute_message_size(thrughput_mode: str,
                        clients: list[dict],
                        arrays: ArrayRecord | None = None,
                        config: ConfigRecord | None = None,
                        metrics: MetricRecord | None = None,
                        variate: ArrayRecord | None = None):
    """Calculate the size of a message in bytes."""
    size = 0
    if arrays is not None:
        arrays = arrays.to_numpy_ndarrays()
        for array in arrays:
            size += array.nbytes
    if variate is not None:
        for array in variate.to_numpy_ndarrays():
            size += array.nbytes
    if config is not None:
        size += len(pickle.dumps(config))
    if metrics is not None:
        size += len(pickle.dumps(metrics))
    
    if thrughput_mode == "train_sent":
        log(INFO, f"Size of the training message sent per client: {size} bytes")
        for client in clients:
            client['train_sent'] = size
    elif thrughput_mode == "evaluate_sent":
        log(INFO, f"Size of the evaluation message sent per client: {size} bytes")
        for client in clients:
            client['evaluate_sent'] = size  
    elif thrughput_mode == "train_received":
        log(INFO, f"Size of the training message received from client with node ID {clients[0]['id']}: {size} bytes")
        for client in clients:
            client['train_received'] = size
    elif thrughput_mode == "evaluate_received":
        log(INFO, f"Size of the evaluation message received from client with node ID {clients[0]['id']}: {size} bytes")
        for client in clients:
            client['evaluate_received'] = size  

def save_results(model: Model, 
                 result: Result, 
                 output_folder: str, 
                 rn_seed: int) -> None:
    # Save the final model
    ndarrays = result.arrays.to_numpy_ndarrays()
    model.set_weights(ndarrays)
    final_model_name = f"model.keras"
    log(INFO, f"Saving final model to disk as {final_model_name}...")
    model.save(output_folder + '/' + final_model_name)

    # Save the training history
    history_filename = f"training_history_{rn_seed}.csv"
    log(INFO, f"Saving training history as {history_filename}...")
    history_file = open(output_folder + '/' + history_filename, 'w', newline='')
    round_fieldnames = ['Round']
    for key in result.evaluate_metrics_clientapp[1].keys():
        round_fieldnames.append(key)
    writer = csv.DictWriter(history_file, fieldnames=round_fieldnames)
    writer.writeheader()
    
    for round in result.evaluate_metrics_clientapp.keys():
        avg_f1 = result.evaluate_metrics_clientapp[round]['avg_f1_score']
        avg_f1_best = result.evaluate_metrics_clientapp[round]['avg_f1_score_best']
        evaluate_metrics = {
            key: f"{value:.5f}" if isinstance(value, float) else f"{value:d}" if isinstance(value, int) else str(value)
            for key, value in result.evaluate_metrics_clientapp[round].items()
        }
        row = {'Round': round if avg_f1 < avg_f1_best else f"*{round}", **evaluate_metrics}
        writer.writerow(row)
    history_file.flush()
    history_file.close()            