#  Derivative work:
#  Copyright 2026 Fondazione Bruno Kessler
# 
#  Original/Previous work:
# 
# Copyright 2025 Flower Labs GmbH. All Rights Reserved.
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
# ==============================================================================
"""Flower message-based Scaffold strategy."""


from collections.abc import Iterable
from logging import INFO
from typing import cast
import numpy as np
import tensorflow as tf

from flwr.app import MessageType
from flwr.serverapp import Grid
from flwr.common import (
    ArrayRecord,
    ConfigRecord,
    Message,
    MetricRecord,
    RecordDict,
    log,
)

from strategies.fedavg4flad import FedAvg4Flad
from .utilities import compute_message_size


# pylint: disable=too-many-instance-attributes
class Scaffold(FedAvg4Flad):
    """Stochastic Controlled Averaging for Federated Learning
    See https://arxiv.org/abs/1910.06378 for details.

    Parameters
    ----------
    client_names : str
        Comma-separated list of client names. Used create named clients and 
        to store per-client metrics.
    fraction_train : float (default: 0.5)
        Fraction of nodes used during training. In case `min_train_nodes`
        is larger than `fraction_train * total_connected_nodes`, `min_train_nodes`
        will still be sampled.
    rn_seed : int (default: 42)
        Random seed for reproducibility of client sampling.
    global_learning_rate : float (default: 1.0)
        Global learning rate used to compute the update on the weights during aggregation.
    variate_arrayrecord_key: str (default: "variate_arrays")
        Key used to store the scaffold variate in the ArrayRecord sent to clients and in the client state. 
    """

    # pylint: disable=too-many-arguments,too-many-positional-arguments
    def __init__(
        self,
        client_names: str,
        fraction_train: float = 0.5,
        rn_seed: int = 42,
        global_learning_rate: float = 1.0,
        variate_arrayrecord_key: str = "variate_arrays",
    ) -> None:
        super().__init__(
            client_names=client_names,
            fraction_train=fraction_train,
            rn_seed=rn_seed,
        )
        self.global_learning_rate = global_learning_rate
        self.variate_arrayrecord_key = variate_arrayrecord_key
        self.global_variate = []
        self.current_arrays = None

    def summary(self) -> None:
        """Log summary configuration of the strategy."""
        log(INFO, "\t├──> Scaffold settings:")
        log(INFO, "\t\t└── Global learning rate: '%s'", self.global_learning_rate)   
        super().summary()


    def configure_train(
        self, server_round: int, arrays: ArrayRecord, config: ConfigRecord, grid: Grid
    ) -> Iterable[Message]:
        """Configure the next round of federated training."""

        # If not yet done, map clients to node IDs in the grid.
        if not self.client2id:
            self._map_clients(grid)

        # Do not configure federated train if fraction_train is 0.
        if self.fraction_train == 0.0:
            return []

        if self.current_arrays is None:
            self.global_variate = [np.zeros(w.shape, dtype=w.dtype) for w in arrays.to_numpy_ndarrays()] 
            for client in self.clients:
                client[self.variate_arrayrecord_key] = [np.zeros(w.shape, dtype=w.dtype) for w in arrays.to_numpy_ndarrays()] 

        # Current model before next training and aggregation
        self.current_arrays = arrays

        # Sample nodes
        num_nodes = int(len(list(grid.get_node_ids())) * self.fraction_train)
        sample_size = max(num_nodes, self.min_train_nodes)

        # Keep stable custom selection logic from FedAvg4Flad
        self._select_clients(sample_size)
        log(
            INFO,
            "configure_train: Sampled %s nodes (out of %s)",
            len(self.participants),
            len(self.clients),
        )

        # Always inject current server round
        config["server_round"] = server_round

        # Send model weights and scaffold variate as two ArrayRecords.
        variate_record = ArrayRecord(self.global_variate)
        record = RecordDict(
            {
                self.arrayrecord_key: arrays,
                self.variate_arrayrecord_key: variate_record,
                self.configrecord_key: config,
            }
        )

        # Increment rounds for sampled clients
        for client in self.participants:
            client['rounds'] += 1

        compute_message_size(thrughput_mode="train_sent", 
                             clients=self.participants, 
                             arrays=arrays, 
                             config=config,
                             variate=variate_record)

        return self._construct_messages(
            record,
            [client['id'] for client in self.participants],
            MessageType.TRAIN,
        )
    
    def _compute_variate(self, participants):
        # Compute global variate 
        new_variate = [np.zeros(w.shape, dtype=w.dtype) for w in self.global_variate] 
        for client in participants: 
            for i, delta in enumerate(client[self.variate_arrayrecord_key]): 
                new_variate[i] += delta 
        new_variate[:] = [variate / len(self.clients) for variate in new_variate] 
        self.global_variate[:] = [g + new for g, new in zip(self.global_variate, new_variate)] 
    
    def aggregate_train(
        self,
        server_round: int,
        replies: Iterable[Message],
    ) -> tuple[ArrayRecord | None, MetricRecord | None]:
        """Aggregate model weights and compute client scores.
        """

        if self.current_arrays is None:
            raise ValueError("Current_arrays is not initialized before aggregate_train")

        # Collect model weights and training metrics from all clients 
        metric_record = MetricRecord({})
        for reply in replies:
            if reply.has_error():
                raise ValueError(f"Error in training message from client with node ID {reply.metadata.src_node_id}: {reply.error}")
            client = next((c for c in self.clients if c['id'] == reply.metadata.src_node_id), None)
            if client is None:
                raise ValueError(f"Received evaluation message from unknown client with node ID {reply.metadata.src_node_id}")
            client[self.arrayrecord_key] = reply.content[self.arrayrecord_key]
            client[self.weighted_by_key] = reply.content['metrics'][self.weighted_by_key]
            # Read variate for this client
            client[self.variate_arrayrecord_key] = [variate for variate in reply.content[self.variate_arrayrecord_key].to_numpy_ndarrays()]
            compute_message_size(thrughput_mode="train_received",
                                 clients=[client],
                                 arrays=reply.content[self.arrayrecord_key],
                                 metrics=reply.content['metrics'],
                                 variate=reply.content[self.variate_arrayrecord_key])      
            client['local_f1'] = reply.content['metrics']['local_f1']
            client['local_training_time'] = reply.content['metrics']['local_training_time']
            # Note that we do not aggregate the metrics but just store them in a per-client manner.                    
            for key, value in reply.content['metrics'].items():
                metric_record[f"{client['name']}_{key}"] = cast(float, value)

        updates_on_weights = []
        new_weights = []
        for weights in self.current_arrays.to_numpy_ndarrays():
            new_weights.append(tf.identity(weights)) 
            updates_on_weights.append(np.zeros(weights.shape,dtype=weights.dtype))

        weights_size = len(self.current_arrays.to_numpy_ndarrays())
        total_samples = 0
        for client in self.participants:
            total_samples += client[self.weighted_by_key]
            client_weights = client[self.arrayrecord_key].to_numpy_ndarrays()
            for weight_index in range(weights_size):
                updates_on_weights[weight_index] += (client_weights[weight_index] - self.current_arrays.to_numpy_ndarrays()[weight_index]) * client[self.weighted_by_key] 
        
        updates_on_weights[:] = [(updates_on_weights[i] * self.global_learning_rate/ total_samples) for i in range(len(updates_on_weights))] 
        aggregated_weights = [new_weights[i] + updates_on_weights[i] for i in range(len(new_weights))] 

        aggregated_weights__arrays = [w.numpy() if tf.is_tensor(w) else np.asarray(w) for w in aggregated_weights]
        array_record = ArrayRecord(aggregated_weights__arrays)

        # After aggregation compute variate 
        self._compute_variate(self.participants) 

        return array_record, metric_record



