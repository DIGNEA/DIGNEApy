#!/bin/bash
# FILEPATH: /home/amarrero/DIGNEApy/gecco_neural_network_experiment.sh

# Run the Python script 10 times
for ((i=0; i<9; i++))
do  
    echo "Running the experiment for the $i time"
    python3 evolve_nn_full_instances.py
    mkdir results_neural_network_encoder_$i
    mv *.csv results_neural_network_encoder_$i/
    mv *.keras results_neural_network_encoder_$i/
done
