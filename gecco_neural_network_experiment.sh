#!/bin/bash
# FILEPATH: /home/amarrero/DIGNEApy/gecco_neural_network_experiment.sh

# Run the Python script 10 times
for ((i=9; i<=9; i++))
do  
    echo "Running the experiment for the $i time"
    python3 -m digneapy.nn_transformer_gecco_24 $i
done
