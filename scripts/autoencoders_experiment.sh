#!/bin/bash


for encoder in 50 100 500 1000 variable
do
    for ((i=0; i<10; i++))
    do
        echo "Running the experiment with encoder: $encoder for the $i time"
        python3 kp_ns_autoencoders.py $encoder $i
    done
done