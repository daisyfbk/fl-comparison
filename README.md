# FL Comparison

The main purpose of this repository is to provide an environment where to test and compare [FLAD](https://github.com/doriguzzi/flad-federated-learning-ddos) with other Federated Learning (FL) strategies. 

In the `./library/strategies` directory, this repository implements 7 strategies (FLAD, FedAvg, FedProx, FedSBS, FedALA, Dafl and Scaffold) integrated with the Flower framework. For each if these strategies the corresponding `server` and `client` apps are provided under the `./nids` directory. 

The dataset for doing the experiments is provided on Hugging Face (see below).

## Creation of the environment

1. Create a python environment (e.g. using `conda`) and install Flower:
    ```bash
    conda create -n fl_comparison python=3.12.2 scikit-learn -y
    conda activate fl_comparison
    pip install -U "flwr[simulation]==1.29.0" 
    ```
1. Install the custom strategy (FLAD) in editable mode:
    ```bash
    pip install -e ./library
    ```      
1. Install the nids common library in editable mode:
    ```bash
    pip install -e ./nids/nids_common
    ```
1. The file `./flwr_config/config.toml` contains a Flower configuration file suitable for executing the experimentation. It must be copied in the `$HOME/.flwr/config.toml` path to be used by the CLI (create that path if it doesn't exist).
1. Download from Hugging Face and reconstruct the dataset in HDF5 format, use the following command:
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
