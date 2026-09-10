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
"""Flower message-based FedProx4Flad strategy."""


from collections.abc import Iterable
from logging import INFO

from flwr.serverapp import Grid
from flwr.common import (
    ArrayRecord,
    ConfigRecord,
    Message,
    log,
)

from strategies.fedavg4flad import FedAvg4Flad

# pylint: disable=too-many-instance-attributes
class FedProx4Flad(FedAvg4Flad):
    """Identical to FedProx but with a custom configure_train() and aggregate_evaluate() method 
    in order to:
    - collect single client metrics 
    - compute the average and standard deviation of the f1_score across clients.
    - aggregate model weights with a simple average (not weighted average)
    Note that all parameters not explicitly mentioned in the __init__() are set to their 
    default values in FedAvg4Flad.
    Not able to obtain the same with evaluate_metrics_aggr_fn().

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
    proximal_mu : float (default: 0.0)
        The weight of the proximal term used in the optimization. 0.0 makes
        this strategy equivalent to FedAvg, and the higher the coefficient, the more
        regularization will be used (that is, the client parameters will need to be
        closer to the server parameters during training).
    """

    # pylint: disable=too-many-arguments,too-many-positional-arguments
    def __init__(
        self,
        client_names: str,
        fraction_train: float = 1.0,
        rn_seed: int = 42,
        proximal_mu: float = 0.0,
    ) -> None:
        super().__init__(
            client_names=client_names,
            fraction_train=fraction_train,
            rn_seed=rn_seed,
        )

        self.proximal_mu = proximal_mu

    def summary(self) -> None:
        """Log summary configuration of the strategy."""
        log(INFO, "\t├──> FedProx4Flad settings:")
        log(INFO, "\t│\t└── Proximal mu: %s", self.proximal_mu)
        super().summary()

    def configure_train(
        self, server_round: int, arrays: ArrayRecord, config: ConfigRecord, grid: Grid
    ) -> Iterable[Message]:
        """Configure the next round of federated training."""
       
        config['proximal_mu'] = self.proximal_mu

        return super().configure_train(server_round, arrays, config, grid)
