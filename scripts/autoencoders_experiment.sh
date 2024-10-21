#!/bin/bash

# Create an array with the numbers
encoders=(100)

# Iterate through the array
for encoder in "${encoders[@]}"
do
    for ((i=0; i<10; i++))
    do
        echo "Running the experiment with encoder: $encoder for the $i time"
        /opt/jupyterhub/bin/python3 ../examples/kp_ae_example.py $encoder $i &
    done
done
