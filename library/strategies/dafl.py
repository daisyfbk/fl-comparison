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
"""Flower message-based DAFL strategy."""


from collections.abc import  Iterable
from logging import INFO
from typing import cast
import math
import numpy as np

from flwr.serverapp import Grid

from flwr.common import (
    ArrayRecord,
    ConfigRecord,
    Message,
    MetricRecord,
    log,
)

from strategies.fedavg4flad import FedAvg4Flad
from .utilities import compute_message_size

# pylint: disable=too-many-instance-attributes
class Dafl(FedAvg4Flad):
    """Disparity-Aware Federated Learning strategy. It aggregates updates only from clients that 
    achieve sufficiently high accuracy on their local data. In addition to the filtering 
    procedure, DAFL also modifies the aggregation strategy by giving more weight to the 
    better-performing local models.
    Implementation based on https://ieeexplore.ieee.org/document/10032055

    Parameters
    ----------
    client_names : str
        Comma-separated list of client names. Used create named clients and 
        to store per-client metrics.
    dafl_threshold : float (default: 0.6)
        The threshold on the validation accuracy below which clients do not send model weights to the server.
    weighted_by_key : str (default: "num-examples")
        The key within each MetricRecord whose value is used as the weight when
        computing weighted averages for ArrayRecords.
    """

    # pylint: disable=too-many-arguments,too-many-positional-arguments
    def __init__(
        self,
        client_names: str,
        dafl_threshold: float = 0.6,
    ) -> None:
        super().__init__(
            client_names=client_names
        )
        self.dafl_threshold = dafl_threshold

    def summary(self) -> None:
        """Log summary configuration of the strategy."""
        log(INFO, "\t├──> DAFL settings:")
        log(INFO, "\t\t└── DAFL threshold: '%s'", self.dafl_threshold)
        super().summary()

    def _select_clients(self, 
                        sample_size,
                        ):

        self.participants =  self.clients

    def configure_train(
        self, server_round: int, arrays: ArrayRecord, config: ConfigRecord, grid: Grid
    ) -> Iterable[Message]:
        """Configure the next round of federated training."""

        config["dafl_threshold"] = self.dafl_threshold
        return super().configure_train(server_round, arrays, config, grid)
    
    def aggregate_train(
        self,
        server_round: int,
        replies: Iterable[Message],
    ) -> tuple[ArrayRecord | None, MetricRecord | None]:
        """Aggregate model weights using method proposed by DAFL.
           This function also collects training metrics from all clients without aggregating them.
        """

        # Collect model weights and training metrics from all clients 
        metric_record = MetricRecord({})
        accepted_clients = []
        for reply in replies:
            if reply.has_error():
                raise ValueError(f"Error in training message from client with node ID {reply.metadata.src_node_id}: {reply.error}")
            client = next((c for c in self.clients if c['id'] == reply.metadata.src_node_id), None)
            if client is None:
                raise ValueError(f"Received evaluation message from unknown client with node ID {reply.metadata.src_node_id}")
            if self.arrayrecord_key not in reply.content or reply.content[self.arrayrecord_key] is None:
                # This is the case when val_acc < dafl_threshold.
                log(INFO, f"Client with {client['name']} did not send model weights.")
                continue
            client[self.arrayrecord_key] = reply.content[self.arrayrecord_key]
            client[self.weighted_by_key] = reply.content['metrics'][self.weighted_by_key]
            client['accuracy'] = reply.content['metrics']['val_acc']
            accepted_clients.append(client)
            compute_message_size(thrughput_mode="train_received",
                                 clients=[client],
                                 arrays=reply.content[self.arrayrecord_key],
                                 metrics=reply.content['metrics'])        
            client['local_f1'] = reply.content['metrics']['local_f1']
            client['local_training_time'] = reply.content['metrics']['local_training_time']
            # Note that we do not aggregate the metrics but just store them in a per-client manner.
            for key, value in reply.content['metrics'].items():
                metric_record[f"{client['name']}_{key}"] = cast(float, value)

        samples = [client[self.weighted_by_key] for client in accepted_clients]
        lambdas = [math.exp(client['accuracy']) for client in accepted_clients]
        total_samples = sum(samples)
        total_lambda = sum(lambdas) 
        coeffs = [s/total_samples * l/total_lambda for s,l in zip(samples, lambdas)]
        total_coeffs = sum(coeffs) 

        # Initialize aggregated weights with zeros. Use the shape of the weights from the first client 
        # to determine the shape of the aggregated weights: all shapes should be the same across clients, 
        # so it does not matter which client we use for this.
        aggregated_weights = []
        for weights in accepted_clients[0][self.arrayrecord_key].to_numpy_ndarrays():
            aggregated_weights.append(np.zeros(weights.shape, dtype=weights.dtype))

        for i, client in enumerate(accepted_clients):
            client_weights = client[self.arrayrecord_key].to_numpy_ndarrays()
            for weight_index in range(len(aggregated_weights)):
                aggregated_weights[weight_index] += client_weights[weight_index] * coeffs[i]/total_coeffs

        array_record = ArrayRecord(aggregated_weights)

        return array_record, metric_record
