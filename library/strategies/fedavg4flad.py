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
"""Flower message-based FedAvg4Flad strategy."""


from collections.abc import  Iterable
from logging import INFO
from typing import cast
import random
import numpy as np

from flwr.serverapp import Grid
from flwr.app import MessageType
from flwr.common import (
    Array,
    NDArray,
    ArrayRecord,
    ConfigRecord,
    Message,
    MetricRecord,
    RecordDict,
    log,
)

from flwr.serverapp.strategy import FedAvg
from flwr.serverapp.strategy.strategy_utils import sample_nodes
from .utilities import compute_message_size, reset_round_client_metrics

# pylint: disable=too-many-instance-attributes
class FedAvg4Flad(FedAvg):
    """Identical to Flower FedAvg but with a custom configure_train() and aggregate_evaluate() methods 
    in order to:
    - collect single client metrics
    - compute the average and standard deviation of the f1_score across clients
    - aggregate model weights with a simple average (not weighted average)
    Note that fraction_evaluate is set to 1.0.
    Not able to obtain the same with evaluate_metrics_aggr_fn() and train_metrics_aggr_fn().

    Parameters
    ----------
    client_names : str
        Comma-separated list of client names. Used create named clients and 
        to store per-client metrics.
    fraction_train : float (default: 1.0)
        Fraction of nodes used during training. In case `min_train_nodes`
        is larger than `fraction_train * total_connected_nodes`, `min_train_nodes`
        will still be sampled.
    rn_seed : int (default: 42)
        Random seed for reproducibility of client sampling.    
    weighted_by_key : str (default: "samples")
        The key within each MetricRecord whose value is used as the weight when
        computing weighted averages for ArrayRecords.
    arrayrecord_key : str (default: "arrays")
        Key used to store the ArrayRecord when constructing Messages.
    configrecord_key : str (default: "config")
        Key used to store the ConfigRecord when constructing Messages.
    """

    # pylint: disable=too-many-arguments,too-many-positional-arguments
    def __init__(
        self,
        client_names: str,
        fraction_train: float = 1.0,
        rn_seed: int = 42,
        weighted_by_key: str = "samples",
        arrayrecord_key: str = "arrays",
        configrecord_key: str = "config",
    ) -> None:
        super().__init__(
            fraction_train=fraction_train,
            fraction_evaluate=1.0,
            min_train_nodes=2,
            min_evaluate_nodes=2,
            min_available_nodes=2,
            weighted_by_key=weighted_by_key,
            arrayrecord_key=arrayrecord_key,
            configrecord_key=configrecord_key,
            train_metrics_aggr_fn=None,
            evaluate_metrics_aggr_fn=None,
        )

        clients = []
        for name in client_names.split(","):
            client = {}
            client['name'] = name 
            client['f1_score'] = 0 
            client['f1_score_best'] = 0
            client['rounds'] = 0
            client['train_sent'] = 0
            client['evaluate_sent'] = 0
            client['train_received'] = 0
            client['evaluate_received'] = 0
            client['local_f1'] = -1
            client['local_training_time'] = -1
            clients.append(client)

        self.clients = clients
        self.best_average_f1 = 0.
        self.best_std_f1 = 0.
        self.participants = []
        self.client2id = False
        self.rng = random.Random(rn_seed)

    def summary(self) -> None:
        """Log summary configuration of the strategy."""
        log(INFO, "\t├──> Clients participating to the federation:")
        for client in self.clients:
            log(INFO, f"\t│\t└── {client['name']}")
        super().summary()

    def _map_clients(self, grid: Grid):
        """Query clients so as to map names to node IDs in the grid."""

        node_ids = list(grid.get_node_ids())
        msgs = []
        for nid in node_ids:
            request = ConfigRecord({"request": "client_info"})
            msgs.append(
                Message(
                    content=RecordDict({self.configrecord_key: request}),
                    message_type="query.info",
                    dst_node_id=nid,
                )
            )

        replies = grid.send_and_receive(msgs, timeout=30.0)
        for reply in replies:
            if reply.has_error():
                raise ValueError(f"Error in evaluation message from client with node ID {reply.metadata.src_node_id}: {reply.error}")
            client = next((c for c in self.clients if c['name'] == reply.content[self.configrecord_key]['name']), None)
            if client is None:
                raise ValueError(f"Received evaluation message from unknown client with node ID {reply.metadata.src_node_id}")
            client['id'] = reply.metadata.src_node_id 
        self.client2id = True    

    # Select the clients participating in the current round 
    def _select_clients(self, 
                        sample_size,
                        ):

        self.participants.clear() 
        self.participants = self.rng.sample(self.clients, sample_size)
        log(INFO, f"All clients: {[client['name'] for client in self.clients]}")
        log(INFO, f"Selected clients for this round: {[client['name'] for client in self.participants]}")
    
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

        # Sample nodes
        num_nodes = int(len(list(grid.get_node_ids())) * self.fraction_train)
        sample_size = max(num_nodes, self.min_train_nodes)

        # I sample nodes in this way because my nodes are pets, not cattle.
        self._select_clients(sample_size)
        log(
            INFO,
            "configure_train: Sampled %s nodes (out of %s)",
            len(self.participants),
            len(self.clients),
        )
        # Always inject current server round
        config["server_round"] = server_round

        # Construct messages
        record = RecordDict(
            {self.arrayrecord_key: arrays, self.configrecord_key: config}
        )

        # Increment the rounds for the sampled clients
        for client in self.participants:
            client['rounds'] += 1

        compute_message_size(thrughput_mode="train_sent", 
                             clients=self.participants, 
                             arrays=arrays, 
                             config=config)

        return self._construct_messages(record, 
                                        [client['id'] for client in self.participants], 
                                        MessageType.TRAIN)

    def aggregate_train(
        self,
        server_round: int,
        replies: Iterable[Message],
    ) -> tuple[ArrayRecord | None, MetricRecord | None]:
          
        metric_record = MetricRecord({})          
        for reply in replies:
            if reply.has_error():
                raise ValueError(f"Error in training message from client with node ID {reply.metadata.src_node_id}: {reply.error}")
            client = next((c for c in self.clients if c['id'] == reply.metadata.src_node_id), None)
            if client is None:
                raise ValueError(f"Received training message from unknown client with node ID {reply.metadata.src_node_id}")
            client[self.arrayrecord_key] = reply.content[self.arrayrecord_key]
            client[self.weighted_by_key] = reply.content['metrics'][self.weighted_by_key]
            compute_message_size(thrughput_mode="train_received",
                                 clients=[client],
                                 arrays=reply.content[self.arrayrecord_key],
                                 metrics=reply.content['metrics'])
            client['local_f1'] = reply.content['metrics']['local_f1']
            client['local_training_time'] = reply.content['metrics']['local_training_time']
            # Note that we do not aggregate the metrics but just store them in a per-client manner.            
            for key, value in reply.content['metrics'].items():
                metric_record[f"{client['name']}_{key}"] = cast(float, value)

        aggregated_np_arrays: dict[str, NDArray] = {}
        total_samples = 0
        for client in self.participants:
            total_samples += client[self.weighted_by_key]
            for key, value in client[self.arrayrecord_key].items():
                if key not in aggregated_np_arrays:
                    aggregated_np_arrays[key] = value.numpy() * client[self.weighted_by_key] 
                else:
                    aggregated_np_arrays[key] += value.numpy() * client[self.weighted_by_key]
        
        for key in aggregated_np_arrays:
            aggregated_np_arrays[key] = aggregated_np_arrays[key] /total_samples

        return ArrayRecord(
            {k: Array(np.asarray(v)) for k, v in aggregated_np_arrays.items()}
        ), metric_record        
    
    def configure_evaluate(
        self, server_round: int, arrays: ArrayRecord, config: ConfigRecord, grid: Grid
    ) -> Iterable[Message]:
        
        # Calculate the size of the message sent.
        compute_message_size(thrughput_mode="evaluate_sent",
                             clients=self.clients,
                             arrays=arrays, 
                             config=config)
        
        return super().configure_evaluate(server_round, arrays, config, grid)

    def aggregate_evaluate(
        self,
        server_round: int,
        replies: Iterable[Message],
    ) -> MetricRecord | None:
        """Aggregate MetricRecords in the received Messages.
           The aggregation is not weighted but it is a simple average of the model weights of all clients.
           Single client evaluation metrics are also collected.
        """

        # Collect evaluation metrics from all clients 
        metric_record = MetricRecord({})
        
        all_clients_f1_scores = []
        total_rounds = 0

        for reply in replies:
            if reply.has_error():
                raise ValueError(f"Error in evaluation message from client with node ID {reply.metadata.src_node_id}: {reply.error}")
            client = next((c for c in self.clients if c['id'] == reply.metadata.src_node_id), None)
            if client is None:
                raise ValueError(f"Received evaluation message from unknown client with node ID {reply.metadata.src_node_id}")
            if 'f1_score' not in reply.content['metrics']:
                raise ValueError(f"'f1_score' not found in evaluation metrics from client with node ID {reply.metadata.src_node_id}")
            client['f1_score'] = cast(float, reply.content['metrics']['f1_score'])
            compute_message_size(thrughput_mode="evaluate_received",
                                 clients=[client],
                                 metrics=reply.content['metrics'])
            all_clients_f1_scores.append(client['f1_score'])
            metric_record[f"{client['name']}_f1_score"] = client['f1_score']
            metric_record[f"{client['name']}_rounds"] = client['rounds']
            metric_record[f"{client['name']}_f1_score_best"] = client['f1_score_best']  
            metric_record[f"{client['name']}_train_sent"] = client['train_sent']
            metric_record[f"{client['name']}_evaluate_sent"] = client['evaluate_sent']
            metric_record[f"{client['name']}_train_received"] = client['train_received']
            metric_record[f"{client['name']}_evaluate_received"] = client['evaluate_received']
            metric_record[f"{client['name']}_local_f1"] = client['local_f1']
            metric_record[f"{client['name']}_local_training_time"] = client['local_training_time']
            reset_round_client_metrics(client)
            total_rounds += client['rounds']
        
        # Average f1_score on all clients
        avg = np.average(all_clients_f1_scores)
        std = np.std(all_clients_f1_scores)
        metric_record[f"avg_f1_score"] = avg
        metric_record[f"std_f1_score"] = std
        if avg > self.best_average_f1:
            self.best_average_f1 = avg
            self.best_std_f1 = std  
            for client in self.clients:
                # the f1_score of this client when the average f1_score across clients is the best 
                # one observed so far.
                client['f1_score_best'] = client['f1_score']
                metric_record[f"{client['name']}_f1_score_best"] = client['f1_score_best']  
        metric_record['avg_f1_score_best'] = self.best_average_f1
        metric_record['std_f1_score_best'] = self.best_std_f1
        metric_record['total_rounds'] = total_rounds
        
        # Sort metric record in this way:
        # - avg_f1_score_best
        # - std_f1_score_best
        # - avg_f1_score
        # - std_f1_score
        # - for each client (alphabetical order): f1_score_best, f1_score, rounds, size messages sent/received
        sorted_metric_record = MetricRecord({})
        sorted_metric_record['avg_f1_score_best'] = metric_record['avg_f1_score_best']
        sorted_metric_record['std_f1_score_best'] = metric_record['std_f1_score_best']
        sorted_metric_record['avg_f1_score'] = metric_record['avg_f1_score']
        sorted_metric_record['std_f1_score'] = metric_record['std_f1_score']
        for client in sorted(self.clients, key=lambda c: c['name']):
            sorted_metric_record[f"{client['name']}_f1_score_best"] = metric_record[f"{client['name']}_f1_score_best"]
            sorted_metric_record[f"{client['name']}_f1_score"] = metric_record[f"{client['name']}_f1_score"]
            sorted_metric_record[f"{client['name']}_rounds"] = metric_record[f"{client['name']}_rounds"]
            sorted_metric_record[f"{client['name']}_train_sent"] = metric_record[f"{client['name']}_train_sent"]
            sorted_metric_record[f"{client['name']}_train_received"] = metric_record[f"{client['name']}_train_received"]
            sorted_metric_record[f"{client['name']}_evaluate_sent"] = metric_record[f"{client['name']}_evaluate_sent"]
            sorted_metric_record[f"{client['name']}_evaluate_received"] = metric_record[f"{client['name']}_evaluate_received"]
            sorted_metric_record[f"{client['name']}_local_f1"] = metric_record[f"{client['name']}_local_f1"]
            sorted_metric_record[f"{client['name']}_local_training_time"] = metric_record[f"{client['name']}_local_training_time"]
        return sorted_metric_record          
    