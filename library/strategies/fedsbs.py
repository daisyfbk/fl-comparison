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
"""Flower message-based FedSBS strategy."""

import math
from collections.abc import  Iterable
from logging import INFO
from typing import cast
import numpy as np
from .utilities import compute_message_size, reset_round_client_metrics

from flwr.common import (
    ArrayRecord,
    ConfigRecord,
    Message,
    MetricRecord,
    RecordDict,
    log,
)

from flwr.server import Grid
from strategies.fedavg4flad import FedAvg4Flad


# pylint: disable=too-many-instance-attributes
class FedSBS(FedAvg4Flad):
    """FedSBS selects clients by estimating their expected improvement in the global model performance
    with the Information Gain (IG) score. Moreover local client models are aggregated with a 
    Federated Averaging with Momentum strategy.

    Implementation based on https://www.sciencedirect.com/science/article/abs/pii/S138912862400183X
    and https://arxiv.org/abs/1909.06335

    Parameters
    ----------
    client_names : str
        Comma-separated list of client names. Used create named clients and 
        to store per-client metrics.
    rounds : int
        Total number of training rounds. Used to compute the decay of epsilon and temperature.    
    fraction_train : float (default: 1.0)
        Fraction of nodes used during training.    
    proximal_mu : float (default: 0.5)
        as in FedProx
    epsilon : float (default: 1.0)
        Initial value of epsilon for epsilon-greedy selection. Epsilon is decayed at each round with 
        a multiplicative factor eta, computed as eta = (epsilon_min)^(1/rounds).
    epsilon_min : float (default: 0.1)
        Minimum value of epsilon for epsilon-greedy selection. Epsilon is decayed at each round with 
        a multiplicative factor eta, computed as eta = (epsilon_min)^(1/rounds).
    temperature : float (default: 5.0)
        Temperature parameter for the Boltzmann distribution used as blocking factor in client selection. 
    beta_momentum : float (default: 0.8)
        Momentum parameter for the FedSBS aggregation method.
    server_learning_rate : float (default: 1.0)
        Learning rate for the FedSBS aggregation method. It is used to compute the momentum as 
        momentum = beta * momentum + learning_rate * update_on_weights.     
    rn_seed : int (default: 42)
        Random seed for reproducibility of client sampling.  
    """

    # pylint: disable=too-many-arguments,too-many-positional-arguments
    def __init__(
        self,
        client_names: str,
        rounds: int,
        fraction_train: float = 1.0,
        proximal_mu: float = 0.5,
        epsilon: float = 1.0,
        epsilon_min: float = 0.1,
        temperature: float = 5.0,
        beta_momentum: float = 0.9,
        server_learning_rate: float = 1.0,
        rn_seed: int = 42,
    ) -> None:
        super().__init__(
            client_names=client_names,
            fraction_train=fraction_train,
            rn_seed=rn_seed,
        )
        self.proximal_mu = proximal_mu
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.temperature = temperature
        self.beta_momentum = beta_momentum
        self.server_learning_rate = server_learning_rate
        self.eta = self.epsilon_min ** (1/rounds)
        for client in self.clients:
            client['fedsbs_score'] = 0. 

        self.current_arrays = None    

    def summary(self) -> None:
        """Log summary configuration of the strategy."""
        log(INFO, "\t├──> FedSBS settings:")
        log(INFO, "\t\t└── FedSBS mu: '%s'", self.proximal_mu)   
        log(INFO, "\t\t└── FedSBS epsilon: '%s'", self.epsilon)
        log(INFO, "\t\t└── FedSBS epsilon min: '%s'", self.epsilon_min)
        log(INFO, "\t\t└── FedSBS temperature: '%s'", self.temperature)
        log(INFO, "\t\t└── FedSBS beta momentum: '%s'", self.beta_momentum)
        log(INFO, "\t\t└── FedSBS server learning rate: '%s'", self.server_learning_rate)
        super().summary()

    def _select_participant(self, 
                           round_counts, 
                           client_scores, 
                           selection_type):
        """ Select a participant with the given selection type (random or greedy) and using a Boltzmann 
        distribution as blocking factor (the higher the round count, the lower the probability of being 
        selected). If the participant is blocked, a new participant is selected with the same selection type.
        """
        
        if selection_type == 'random':
            i = self.rng.choice(list(client_scores.keys()))
        else:
            # Get the client with highest score 
            i = max(client_scores, key=client_scores.get) 

        blocked = True 
        tries = 1 
        while blocked:
            if round_counts[i] == 0: 
                blocked = False 
                break 

            # Bolzmann probability: the higher the round count, the lower the probability of being selected.    
            p = math.exp(-round_counts[i]/self.temperature)
            
            if self.rng.uniform(0., 1.) < p:
                # Accepted 
                blocked = False 
                break 
            elif tries >= len(client_scores): 
                # Too many tries => random choice
                log(INFO, f"FEDSBS: Too many tries, random choice {i}") 
                i = self.rng.choice(list(client_scores.keys()))
                blocked = False 
            else:
                # New try 
                if selection_type == "random":
                    i = self.rng.choice(list(client_scores.keys()))
                else:
                    # Get the next greedy element 
                    i = list(sorted(client_scores.items(), key=lambda item: item[1], reverse=True))[tries][0] 
                tries += 1 

        return i 

    def _epsilon_greedy_selection(self, 
                                  sample_size,
                                  round_counts, 
                                  client_scores):
        """"Return indexes of participants chosen with epsilon-greedy method."""

        participants = []
        # If the selection is random we prefer to select never trained client 
        never_selected = [i for i, c in round_counts.items() if c == 0] 

        # The first time we choose randomly (no computed scores)
        if len(never_selected) == len(round_counts.keys()):
            return self.rng.sample(never_selected, sample_size) 

        for _ in range(sample_size):
            if self.rng.uniform(0, 1) < self.epsilon:
                # Random selection 
                if len(never_selected) > 0:
                    j = self.rng.choice(never_selected)
                    never_selected.remove(j) 
                else:
                    j = self._select_participant(round_counts, client_scores, "random")
            else:
                # Greedy selection 
                j = self._select_participant(round_counts, client_scores, "greedy")

            del round_counts[j] 
            del client_scores[j] 
            participants.append(j)

        return participants

    def _select_fedsbs(self, sample_size):
        round_counts = { i:client['rounds'] for i, client in enumerate(self.clients) }
        client_scores = { i:client['fedsbs_score'] for i, client in enumerate(self.clients)}
        idxs = self._epsilon_greedy_selection(sample_size,round_counts, client_scores)

        return [self.clients[i] for i in idxs]
    
    def _select_clients(self, 
                        sample_size,
                        ):
        log(INFO,f"FedSBS scores are: { {client['name']:client['fedsbs_score'] for client in self.clients} }") 
        self.participants.clear() 
        self.participants = self._select_fedsbs(sample_size)
        
        self.epsilon *= self.eta
        
        # TODO: understand if needed!!!! Guess NO.
        # self.fedsbs_temperature *= self.cool 

        log(INFO, f"Selected clients for this round: {[client['name'] for client in self.participants]}")
    
    def configure_train(
        self, server_round: int, arrays: ArrayRecord, config: ConfigRecord, grid: Grid
    ) -> Iterable[Message]:
        """Configure the next round of federated training."""
     
        # Last model before next training and aggregation.
        # Initialize aggregation momentum to zeros the first time.
        if self.current_arrays is None:
            self.momentum = [np.zeros(w.shape, dtype=w.dtype) for w in arrays.to_numpy_ndarrays()]
        self.current_arrays = arrays

        config['proximal_mu'] = self.proximal_mu
        return super().configure_train(server_round, arrays, config, grid)
    
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
            client['fedsbs_score'] = reply.content['metrics']['fedsbs_score']
            compute_message_size(thrughput_mode="train_received",
                                 clients=[client],
                                 arrays=reply.content[self.arrayrecord_key],
                                 metrics=reply.content['metrics'])   
            client['local_f1'] = reply.content['metrics']['local_f1']
            client['local_training_time'] = reply.content['metrics']['local_training_time']   
            # Note that we do not aggregate the metrics but just store them in a per-client manner.               
            for key, value in reply.content['metrics'].items():
                metric_record[f"{client['name']}_{key}"] = cast(float, value)

        # Aggregate model weights using Federated Averaging with Momentum (FedAvgM) [Hsu et al., 2019]
        # TODO: it is not possible to verify if the following implementation is correct since the paper 
        # does not provide enough information. Implementation has been taken from the literature.
    
        updates_on_weights = []

        for weights in self.current_arrays.to_numpy_ndarrays():
            updates_on_weights.append(np.zeros(weights.shape, dtype=weights.dtype))

        weights_list_size = len(updates_on_weights)
        total_samples = 0
        for client in self.participants:
            total_samples += client[self.weighted_by_key]
            client_weights = client[self.arrayrecord_key].to_numpy_ndarrays()
            for weight_index in range(weights_list_size):
                # Note that it is the opposite of gradient (old - new)
                updates_on_weights[weight_index] += (self.current_arrays.to_numpy_ndarrays()[weight_index] - client_weights[weight_index]) * client[self.weighted_by_key] 
        
        updates_on_weights[:] = [(updates_on_weights[i] / total_samples) for i in range(len(updates_on_weights))] 

        self.momentum[:] = [self.beta_momentum * m + u for m, u in zip(self.momentum, updates_on_weights)] 
        aggregated_weights = [old - self.server_learning_rate * mom for old, mom in zip(self.current_arrays.to_numpy_ndarrays(), self.momentum)] 

        array_record = ArrayRecord(aggregated_weights)

        return array_record, metric_record
