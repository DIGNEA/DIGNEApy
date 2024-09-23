#!/bin/bash
# FILEPATH: /home/amarrero/DIGNEApy/gecco_neural_network_experiment.sh

# Run the Python script 10 times
for ((i=0; i<9; i++))
do  
    echo "Running the experiment for the $i time"
    python3 -m digneapy.nn_transformer_gecco_24
    mkdir results_gecco_nn_run_$i
    mv *.csv results_gecco_nn_run_$i/
    mv *.keras results_gecco_nn_run_$i/
done
