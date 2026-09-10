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
"""Flower message-based FedALA strategy."""


from collections.abc import Iterable
from logging import INFO
from tkinter import Grid

from flwr.common import (
    ArrayRecord,
    ConfigRecord,
    Message,
    log,
)

from flwr.serverapp import Grid

from strategies.fedavg4flad import FedAvg4Flad


# pylint: disable=too-many-instance-attributes
class FedALA(FedAvg4Flad):
    """FedALA: Adaptive Local Aggregation for Personalized Federated Learning.
    See https://arxiv.org/abs/2212.01197.
    This strategy is the same as FedAvg4Flad with just some more parameters exchanged
    between clients and server. 

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
    ala_trainable_layers : int (default: 1)
        Number of trainable layers in the ALA model. The remaining layers are frozen.
    ala_sample_ratio : float (default: 0.8)   
            The ratio of local training data to be used for training the ALA model. 
    """

    # pylint: disable=too-many-arguments,too-many-positional-arguments
    def __init__(
        self,
        client_names: str,
        fraction_train: float = 1.0,
        rn_seed: int = 42,
        ala_trainable_layers: int = 1,
        ala_sample_ratio: float = 0.8,
    ) -> None:
        super().__init__(
            client_names=client_names,
            fraction_train=fraction_train,
            rn_seed=rn_seed,
        )
        self.ala_trainable_layers = ala_trainable_layers
        self.ala_sample_ratio = ala_sample_ratio

    def summary(self) -> None:
        """Log summary configuration of the strategy."""
        super().summary()
        log(INFO, f"\t├──> FedALA-specific parameters:")
        log(INFO, f"\t│\t├──> ala_trainable_layers: {self.ala_trainable_layers}")
        log(INFO, f"\t│\t└──> ala_sample_ratio: {self.ala_sample_ratio}")

    def configure_train(
        self, server_round: int, arrays: ArrayRecord, config: ConfigRecord, grid: Grid
    ) -> Iterable[Message]:
        """Configure the next round of federated training."""

        # Send the two fedala-specific parameters to the clients.
        config["ala_trainable_layers"] = self.ala_trainable_layers
        config["ala_sample_ratio"] = self.ala_sample_ratio
        
        return super().configure_train(server_round, arrays, config, grid)