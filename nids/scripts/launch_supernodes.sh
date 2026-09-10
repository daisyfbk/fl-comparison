#!/bin/bash
#
# Copyright 2025 Fondazione Bruno Kessler.
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

# Launch all the supernodes processes, one for each client
# Get the name of the clients from the command line
# Then loop over the clients and launch a supernode for each of them,
# increasing the clientappio-api-address port for each supernode and using a 
# different data folder for each supernode.

# Parse command line arguments
if [ $# -eq 0 ]; then
    echo "Usage: $0 <client1> <client2> ..."
    exit 1
fi

# First of all, kill all the existing supernodes processes, to avoid conflicts with the new ones
pkill -f "flower-supernode" || true
while pgrep -f "flower-supernode" > /dev/null; do echo "still some process to be killed"; sleep 1; done

# Create logs directory if it doesn't exist
mkdir -p "./logs/clients"
rm ./logs/clients/*.log 2>/dev/null || true

DATA_FOLDER="../../dataset/DOS201X_highly_unbalanced"

# Loop over the clients and launch a supernode for each of them
port=9094
for i in "${@:1}"; do
    client_name=$(echo "$i" | tr -d '"')
    echo "Launching supernode for client $client_name"
    data_folder="${DATA_FOLDER}/${client_name}"
    log_file="./logs/clients/${client_name}.log"
    rm $log_file 2>/dev/null || true
    FLWR_LOG_LEVEL=DEBUG flower-supernode --insecure --superlink 127.0.0.1:9092  \
    --clientappio-api-address 127.0.0.1:$port \
    --node-config "name=\"$client_name\" data_folder=\"$data_folder\"" > "$log_file" 2>&1 &
    port=$((port + 1))
done
