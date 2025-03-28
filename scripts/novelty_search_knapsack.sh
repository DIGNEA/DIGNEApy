#!/bin/bash

# Create an array with the numbers
#dimensions=(50 100 250 500 1000)

# Iterate through the array
#for dimension in "${dimensions[@]}"
#do
for ((i=0; i<10; i++))
do
	echo "Running the experiment with N: $dimension for the $i time"
        python3 kp_ns_8d_example.py 1000 $i
done
#done
