# Comparison of Federated Learning Strategies for Cybersecurity under Non-IID and Unbalanced Settings
This repository contains the code and experiments for our systematic evaluation of Federated Learning (FL) methods applied to intrusion detection in the context of DDoS attacks. These methods have been specifically designed to tackle heterogeneous FL scenarios, where the training data is non-independent and identically distributed (non-i.i.d) and unbalanced across the clients.

The evaluation includes Federated Averaging (FedAVG), the original FL algorithm proposed by McMahan et al., which is used as a baseline, three general purpose FL algorithms and three FL algorithms specifically designed for cybersecurity tasks.
The full list of algorithms is the following:
- **FedAVG** (B. McMahan, E. Moore, D. Ramage, S. Hampson, and B. A. y Arcas, “Communication-efficient learning of deep networks from decentralized data,” in Artificial intelligence and statistics, 2017.)
- **FedProx** (T. Li, A. K. Sahu, M. Zaheer, M. Sanjabi, A. Talwalkar, and V. Smith, “Federated optimization in heterogeneous networks,” Proceedings of Machine learning and systems, vol. 2, pp. 429–450, 2020.)
- **SCAFFOLD** (S. P. Karimireddy, S. Kale, M. Mohri, S. Reddi, S. Stich, and A. T. Suresh, “SCAFFOLD: Stochastic controlled averaging for federated learning,” in Proc. of International conference on machine learning. PMLR, 2020.)
- **FedALA** (J. Zhang, Y. Hua, H. Wang, T. Song, Z. Xue, R. Ma, and H. Guan, “FedALA: Adaptive Local Aggregation for Personalized Federated Learning,” Proc. of the AAAI Conference on Artificial Intelligence, 2023.)
- **DAFL** (J. Li, X. Tong, J. Liu, and L. Cheng, “An efficient federated learning system for network intrusion detection,” IEEE Systems Journal, vol. 17, no. 2, pp. 2455–2464, 2023.)
- **FedSBS** (H. N. Cunha Neto, J. Hribar, I. Dusparic, N. C. Fernandes, and D. M. Mattos, “FedSBS: Federated-Learning participant-selection method for Intrusion Detection Systems,” Computer Networks, vol. 244, p. 110351, 2024.)
- **FLAD** (R. Doriguzzi-Corin and D. Siracusa, “FLAD: Adaptive Federated Learning for DDoS attack detection,” Computers & Security, vol. 137, p. 103597, 2024.)

The above algorithms are compared in terms of overall duration of the FL process, local training time, network bandwidth overhead and final model accuracy. The goal is to benchmark a diverse set of FL algorithms with respect to their capacity to produce a global model that generalises effectively across heterogeneous client datasets. 

The setup provided in this repository has been implemented to stress-test the algorithms under extreme non-i.i.d. conditions, thereby highlighting their relative strengths and limitations in handling realistic adversarial distributions.

The objective of this work is to understand how different FL training strategies behave in cybersecurity scenarios characterised by challenging client-level heterogeneity and to quantify the resulting trade-offs between detection performance, communication overhead, and training time.

In the `./library/strategies` directory, this repository implements 7 strategies (FLAD, FedAvg, FedProx, FedSBS, FedALA, Dafl and Scaffold) integrated with the Flower framework. For each if these strategies the corresponding `server` and `client` apps are provided under the `./nids` directory. 

## Dataset
The evaluation is performed with recent datasets of DDoS attacks, [IDS2017](https://www.unb.ca/cic/datasets/ids-2017.html), [IDS2018](https://www.unb.ca/cic/datasets/ids-2018.html) and [DDoS2019](https://www.unb.ca/cic/datasets/ddos-2019.html), provided by the Canadian Institute of Cybersecurity of the University of New Brunswick. These datasets contain several days of network activity, provided in the form of packet capture files (pcap). In this work, the pcap files are processed to extract network flows using [LUCX](https://github.com/doriguzzi/LUCX_network_traffic_parser), which aggregates packet-level features into flow-level representations. The final dataset consists of several days of network activity, and includes both benign traffic and 15 different types of DDoS attacks.
For evaluation purposes, the dataset is distributed among the clients in a non-i.i.d. and unbalanced manner: each attack type is assigned to a different client, and the amount of data varies across clients to simulate a realistic FL environment where clients possess heterogeneous data both in terms of distribution and quantity.
The dataset is publicly avaiable on [HuggingFace](https://huggingface.co/datasets/silviocretti/DOS201X_highly_unbalanced).

## Testing environment
Each FL method has been implemented as a Flower strategy. [Flower](https://flower.ai/) is a framework that provides a high-level interface for implementing FL algorithms and managing the communication between the server and clients. The system can be deployed using the Flower Deployment Runtime on a single server machine, with the server and clients running as separate processes. Communication between the server and clients is handled through the gRPC protocol integrated into the Flower framework, making it completely transparent to both the server and client applications.

1. Create a python environment (e.g. using `conda`) and install Flower:
    ```bash
    conda create -n fl_comparison python=3.12.2 scikit-learn -y
    conda activate fl_comparison
    pip install -U "flwr[simulation]==1.29.0" 
    ```
2. Install the custom strategy (FLAD) in editable mode:
    ```bash
    pip install -e ./library
    ```      
3. Install the nids common library in editable mode:
    ```bash
    pip install -e ./nids/nids_common
    ```
4. The file `./flwr_config/config.toml` contains a Flower configuration file suitable for executing the experimentation. It must be copied in the `$HOME/.flwr/config.toml` path to be used by the CLI (create that path if it doesn't exist).
5. Download from Hugging Face and reconstruct the dataset in HDF5 format, use the following command:
   ```bash
   python ./nids/scripts/prepare_dataset.py --repo-id silviocretti/DOS201X_highly_unbalanced
   ```
   By default, the dataset will be saved in `./dataset/DOS201X_highly_unbalanced`. You can also specify a different output folder with the `--output-folder` parameter but recall to keep it coherent the `DATA_FOLDER` variable in `./nids/scripts/launch_supernodes.sh`.

## Execution of the experiments with different strategies

We start a Flower federation with one SuperLink and a number of SuperNodes (i.e. clients). We simplify the deployment having clients (SuperNodes) on the same machine but in a real scenario they should be on different machines.

1. (If not yet done) Activate the conda environment on all the terminals you plan to use:
    ```bash
    conda activate fl_comparison
    ```
1. Change directory to the app folder of the strategy you want to run:
    ```bash
    cd nids/<strategy>
    ``` 
1. Configure your experiment in the `./pyproject.toml` file: user can adjust parameters in the section `[tool.flwr.app.config]`.
1. On a terminal start the SuperLink (server side):
    ```bash
    flower-superlink --insecure    
    ```
1. On another terminal start the SuperNodes (client side). In this experiment, we start 15 SuperNodes, each one with a different client name. The same names must be configured inside the `pyproject.toml` file of the app (see `client_names` parameter).
    ```bash
    ../scripts/launch_supernodes.sh 00-WebDDoS 01-LDAP 02-Portmap 03-DNS 04-UDPLag 05-NTP 06-SNMP 07-SSDP 08-Syn 09-TFTP 10-UDP 11-NetBIOS 12-MSSQL 13-LOIC 14-HOIC
    ```         
1. In another terminal, run the app:
   ```bash
   flwr run . local-deployment --stream
    ```   
    Note: `local-deployment` is specified in the Flower configuration file (`$HOME/.flwr/config.toml` (see above)
1. The output is saved in the `./logs` folder.

## Acknowledgements

If you are using this code for scientific research, please cite the related paper in your manuscript as follows:

Roberto Doriguzzi-Corin, Silvio Cretti, Petr Sabel, Silvio Ranise, "Federated Learning in the Wild: A Comparative Study for Cybersecurity under Non-IID and Unbalanced Settings", 2025. [Online]. Available: https://arxiv.org/abs/2509.17836

This work was supported by Ministero delle Imprese e del Made in Italy (IPCEI Cloud DM 27 giugno 2022 – IPCEI-CL-0000007) and European Union (Next Generation EU).

## License

Copyright 2024 Fondazione Bruno Kessler

Licensed under the Apache License, Version 2.0 (the “License”); you may not use this
file except in compliance with the License. You may obtain a copy of the License
[here](http://www.apache.org/licenses/LICENSE-2.0).

Unless required by applicable law or agreed to in writing, software distributed under
the License is distributed on an “AS IS” BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
either express or implied. See the License for the specific language governing permissions
and limitations under the License.
