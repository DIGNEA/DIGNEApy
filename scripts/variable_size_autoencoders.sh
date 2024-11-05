#!/bin/bash


for dimension in 50 100 500 1000 variable
do
    for ((i=0; i<10; i++))
    do
        echo "Running the experiment with dimension: $dimension for the $i time"
        python3 variable_autoencoder_kp.py $dimension $i
    done
done
