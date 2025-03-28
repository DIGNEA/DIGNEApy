#!/bin/bash

# Create an array with the numbers
dimensions=(50)
encodings=(2)

# Iterate through the array
for dimension in "${dimensions[@]}"
do
    for ((i=0; i<10; i++))
    do
        for encoding in "${encodings[@]}"
        do
            echo "Running the experiment with N: $dimension and encoding: $encoding for the $i time"
            python3 kp_ae_example.py $dimension $encoding $i
        done
    done
done
