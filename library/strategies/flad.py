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
"""Flower message-based FedAvg4Flad strategy."""

from ctypes import Array
import io
from multiprocessing.dummy import Array
import time
from logging import INFO
import math
import numpy as np
from typing import cast

from collections.abc import Iterable
from flwr.common import log, Array, NDArray
from flwr.app import ArrayRecord, ConfigRecord, Message, MetricRecord, RecordDict, MessageType
from flwr.serverapp import Grid
from flwr.serverapp.strategy import Result
from .utilities import compute_message_size, reset_round_client_metrics

class Flad:
    """Flad strategy.

    This implementation is based on paper "FLAD: Adaptive Federated Learning for DDoS attack detection".
    This class is not derived from Strategy because we need to re-implement the start() method to accommodate 
    the FLAD stopping condition.

    Parameters 
    ----------
    client_names : list 
        Comma-separated list of client names. Used create named clients and 
        to store per-client metrics.
    weighted_by_key : str (default: "samples")
        The key within each MetricRecord whose value is used as the weight when
        computing weighted averages for ArrayRecords.
    arrayrecord_key : str (default: "arrays")
        Key used to store the ArrayRecord when constructing Messages.
    configrecord_key : str (default: "config")
        Key used to store the ConfigRecord when constructing Messages.  
    """
     
    def __init__(self, client_names: str, 
                 weighted_by_key: str = "samples",
                 arrayrecord_key: str = "arrays",
                 configrecord_key: str = "config",
                 ) -> None: 
        self.arrayrecord_key = arrayrecord_key
        self.configrecord_key = configrecord_key
        self.weighted_by_key = weighted_by_key
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
            client[self.arrayrecord_key] = None
            clients.append(client)
        
        self.clients = clients
        self.last_average_f1 = 1.0
        self.best_average_f1 = 0.0
        self.best_std_f1 = 0.0
        self.participants = []
        self.client2id = False
    
    def summary(self) -> None:
        """Log summary configuration of the strategy."""

        log(INFO, "\t├──> Clients participating to the federation:")
        for client in self.clients:
            log(INFO, f"\t│\t└── {client['name']}")
        log(INFO, "\t└──> Keys in records:")
        log(INFO, f"\t\t├── ArrayRecord key: '{self.arrayrecord_key}'")
        log(INFO, f"\t\t└── ConfigRecord key: '{self.configrecord_key}'")
        log(INFO, f"\t\t└── Weighted by key: '{self.weighted_by_key}'")

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

    def _construct_messages(self, record: RecordDict, 
                            message_type: str,
                            )-> Iterable[Message]:
        messages = []
        for client in self.participants:  
            if message_type == MessageType.TRAIN:
                this_client_config =  record[self.configrecord_key].copy()
                this_client_config['epochs'] = client['epochs']
                this_client_config['steps_per_epoch'] = client['steps_per_epoch']
                record[self.configrecord_key] = this_client_config
                # Count training rounds 
                client['rounds'] += 1
                client_record = RecordDict({
                    self.arrayrecord_key: record[self.arrayrecord_key],
                    self.configrecord_key: this_client_config
                })
                # Calculate the size of the message sent. Did here because client_record is updated.
                compute_message_size(thrughput_mode="train_sent", 
                            clients=[client], 
                            arrays=client_record[self.arrayrecord_key], 
                            config=client_record[self.configrecord_key]) 
            else:
                client_record = record    
            message = Message(
                content=client_record,
                message_type=message_type,
                dst_node_id=client['id'],
            )
            messages.append(message)
           
        return messages

    def _select_clients(self, 
                        all_clients: bool = False,
                        ):
        """Select the clients participating in the current round based on their F1 Score."""

        self.participants.clear()
        for client in self.clients: 
            if all_clients or client['f1_score'] <= self.last_average_f1: 
                self.participants.append(client)

    def _scale_linear_bycolumn(
        self,
        rawpoints, 
        mins,
        maxs,
        high = 1.0, 
        low = 0.0):
        rng = maxs - mins
        return high - (((high - low) * (maxs - rawpoints)) / rng) 

    def _update_client_training_parameters(self, 
                                           parameter: str,
                                           min_value: int,
                                           max_value: int, 
                                           ):
        """Compute the training parameters (epochs and steps_per_epoch) for the clients based on their F1 Score."""
        f1_list = []

        for client in self.participants:
            f1_list.append(client['f1_score'])

        if len(set(f1_list)) > 1:
            min_f1_value = min(f1_list)
            max_value = max(min_value + 1, math.ceil(max_value*(1 - min_f1_value))) 
            value_list = max_value + min_value - self._scale_linear_bycolumn(f1_list, 
                                                                       np.min(f1_list), 
                                                                       np.max(f1_list), 
                                                                       high = float(max_value), 
                                                                       low = min_value)
        else: 
            value_list = [max_value] * len(self.participants)

        for client in self.participants:
            client[parameter] = int(value_list[self.participants.index(client)]) 

    def configure_train(self, 
                        arrays: ArrayRecord, 
                        config: ConfigRecord,
                        server_round: int, 
                        grid: Grid) -> Iterable[Message]:
        """Configure the next round of federated training."""
        
        # If not yet done, map clients to node IDs in the grid.Clients are pets.
        # Moreover initialize clients model with the global model received.
        if not self.client2id:
            self._map_clients(grid) 
            for client in self.clients:    
                client[self.arrayrecord_key] = arrays  
            self.client2id = True     

        # Select participants clients based on F1 Score
        self._select_clients(all_clients=False) 
        log(INFO, f"configure_train: all clients: {[client['name'] for client in self.clients]}")
        log(INFO, f"configure_train: selected clients for this round: {[client['name'] for client in self.participants]}") 


        # Update training parameters for the selected clients
        minepochs = config['min_epochs']
        maxepochs = config['max_epochs']
        minsteps = config['min_steps']
        maxsteps = config['max_steps']
        if  isinstance(minepochs, int) and \
            isinstance(maxepochs, int) and \
            isinstance(minsteps, int) and \
            isinstance(maxsteps, int):
            minepochs = int(minepochs)
            maxepochs = int(maxepochs)
            minsteps = int(minsteps)
            maxsteps = int(maxsteps)
        else:
            raise TypeError("Training parameters (min_epochs, max_epochs, min_steps, max_steps) must be integers!")
        
        self._update_client_training_parameters('epochs', 
                                                minepochs, 
                                                maxepochs,
                                                )
        self._update_client_training_parameters('steps_per_epoch', 
                                                minsteps, 
                                                maxsteps,
                                                )
    
        # Now build the messages only for selected clients
        config_record = ConfigRecord({"server_round": server_round,})
        
        record = RecordDict({self.arrayrecord_key: arrays, 
                             self.configrecord_key: config_record})

        return self._construct_messages(record, MessageType.TRAIN)

    def configure_evaluate(self, 
                           arrays: ArrayRecord, 
                           config: ConfigRecord, 
                           grid: Grid) -> Iterable[Message]:
        """Configure the next round of federated evaluation."""

        # Calculate the size of the message sent.
        compute_message_size(thrughput_mode="evaluate_sent",
                             clients=self.clients,
                             arrays=arrays, 
                             config=config)

        # Select all clients for evaluation phase.
        self._select_clients(all_clients=True) 
        log(INFO, f"configure_evaluate: on {len(self.clients)} clients")

        # Construct messages
        record = RecordDict({self.arrayrecord_key: arrays, self.configrecord_key: config})
        return self._construct_messages(record, MessageType.EVALUATE)      

    def aggregate_train(self, 
                        train_replies: Iterable[Message],
                        ) -> tuple[ArrayRecord, MetricRecord]:
        """Aggregate model weights on all clients (not only on the ones that participated in the round).
           The aggregation is not weighted but it is a simple average of the model weights of all clients.
           This function also collects training metrics from all clients without aggregating them.
        """       

        # Collect model weights and training metrics from all clients 
        metric_record = MetricRecord({})
        log(INFO, f"aggregate_train: received {len(train_replies)} replies from clients")
        for reply in train_replies:
            if reply.has_error():
                raise ValueError(f"Error in training message from client with node ID {reply.metadata.src_node_id}: {reply.error}")
            client = next((c for c in self.clients if c['id'] == reply.metadata.src_node_id), None)
            if client is None:
                raise ValueError(f"Received evaluation message from unknown client with node ID {reply.metadata.src_node_id}")
            client[self.arrayrecord_key] = reply.content[self.arrayrecord_key]
            compute_message_size(thrughput_mode="train_received",
                                 clients=[client],
                                 arrays=reply.content[self.arrayrecord_key],
                                 metrics=reply.content['metrics'])
            # Save local_f1 and local_training_time only if they are present. Otherwise we keep -1.
            if 'local_f1' in reply.content['metrics']:
                client['local_f1'] = reply.content['metrics']['local_f1']
            if 'local_training_time' in reply.content['metrics']:
                client['local_training_time'] = reply.content['metrics']['local_training_time']
            # We just collect the metrics without aggregating them.
            for key, value in reply.content['metrics'].items():
                metric_record[f"{client['name']}_{key}"] = cast(float, value)

        aggregated_np_arrays: dict[str, NDArray] = {}
        for client in self.clients:
            for key, value in client[self.arrayrecord_key].items():
                if key not in aggregated_np_arrays:
                    aggregated_np_arrays[key] = value.numpy() 
                else:
                    aggregated_np_arrays[key] += value.numpy() 
        
        for key in aggregated_np_arrays:
            aggregated_np_arrays[key] = aggregated_np_arrays[key] /len(self.clients)

        return ArrayRecord(
            {k: Array(np.asarray(v)) for k, v in aggregated_np_arrays.items()}
        ), metric_record
    
    def aggregate_evaluate(
        self,
        server_round: int,
        replies: Iterable[Message],
    ) -> MetricRecord:
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
            compute_message_size(thrughput_mode="evaluate_received",
                                 clients=[client],
                                 metrics=reply.content['metrics'])            
            client['f1_score'] = cast(float, reply.content['metrics']['f1_score'])
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
    
    def start(
        self,
        grid: Grid,
        initial_arrays: ArrayRecord,
        timeout: float = 3600,
        train_config: ConfigRecord | None = None,
        evaluate_config: ConfigRecord | None = None,
    ) -> Result:
        """Execute the Flad federated learning strategy.

        Runs the complete federated learning workflow, including training and evaluation. 
        This method is customized for Flad strategy bacause Flad does not use a fixed
        number of rounds, but continues until a stopping condition is met.

        Parameters
        ----------
        grid : Grid
            The Grid instance used to send/receive Messages from nodes executing a
            ClientApp.
        initial_arrays : ArrayRecord
            Initial model parameters (arrays) to be used for federated learning.
        timeout : float (default: 3600)
            Timeout in seconds for waiting for node responses.
        train_config : ConfigRecord, optional
            Configuration to be sent to nodes during training rounds.
            If unset, an empty ConfigRecord will be used.
        evaluate_config : ConfigRecord, optional
            Configuration to be sent to nodes during evaluation rounds.
            If unset, an empty ConfigRecord will be used.           

        Returns
        -------
        Results
            Results containing final model arrays and also training metrics, evaluation
            metrics and global evaluation metrics (if provided) from all rounds.
        """
        log(INFO, f"Starting {self.__class__.__name__} strategy:")
        self.summary()
        log(INFO, "")

        # Initialize if None
        train_config = ConfigRecord() if train_config is None else train_config
        evaluate_config = ConfigRecord() if evaluate_config is None else evaluate_config
        result = Result()

        t_start = time.time()

        # Initialize variable before the loop.
        arrays = initial_arrays
        current_round = 0
        stop_counter = 0
        stop_condition = False
        while stop_condition == False:
            current_round += 1
            log(INFO, "")
            log(INFO, "[ROUND %s]", current_round)

            # -----------------------------------------------------------------
            # --- TRAINING (CLIENTAPP-SIDE) -----------------------------------
            # -----------------------------------------------------------------

            # Call strategy to configure training round. Send messages and wait for replies
            train_replies = grid.send_and_receive(
                messages=self.configure_train(
                    arrays,
                    train_config,
                    current_round,
                    grid,
                ),
                timeout=timeout,
            )

            # Aggregate training results on all clients.
            agg_arrays, agg_train_metrics = self.aggregate_train(train_replies)


            # Update server and client models with the aggregated model weights.
            if agg_arrays is not None:
                arrays = agg_arrays
            
            for client in self.clients:
                client[self.arrayrecord_key] = arrays

            # Log training metrics and append to results.
            if agg_train_metrics is not None:
                log(INFO, f"\t└──> Aggregated MetricRecord: {agg_train_metrics}")
                result.train_metrics_clientapp[current_round] = agg_train_metrics

            # -----------------------------------------------------------------
            # --- EVALUATION (CLIENTAPP-SIDE) ---------------------------------
            # -----------------------------------------------------------------

            # Configure evaluation round. Send messages and wait for replies
            evaluate_replies = grid.send_and_receive(
                messages=self.configure_evaluate(
                    arrays,
                    evaluate_config,
                    grid,
                ),
                timeout=timeout,
            )

            # Aggregate evaluate.
            agg_evaluate_metrics = self.aggregate_evaluate(
                current_round,
                evaluate_replies,
            )

            # Log evaluation metrics and append to results.
            if agg_evaluate_metrics is not None:
                log(INFO, f"\t└──> Aggregated MetricRecord: {agg_evaluate_metrics}")
                result.evaluate_metrics_clientapp[current_round] = agg_evaluate_metrics
        
            # Calculate stopping condition and best model.
            self.last_average_f1 = cast(float, agg_evaluate_metrics['avg_f1_score'])
            if self.last_average_f1 < self.best_average_f1:
                stop_counter += 1
            else:
                # Contrary to Flower standard strategy, we keep the best model not the last one.
                result.arrays = arrays
                stop_counter = 0

            log(INFO, f"Current patience counter: {stop_counter}")
            
            # Check stopping condition    
            stop_condition = True if stop_counter > train_config['patience'] else False      

        log(INFO, f"Strategy execution finished in {time.time() - t_start:.2f}s")
        log(INFO, "")
        log(INFO, "Final results:")
        log(INFO, "")
        for line in io.StringIO(str(result)):
            log(INFO, "\t%s", line.strip("\n"))
        log(INFO, "")

        return result          