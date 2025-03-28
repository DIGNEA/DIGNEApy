#!/bin/bash

# Create an array with the numbers
dimensions=(500)
# Iterate through the array
for dimension in "${dimensions[@]}"
do
    for ((i=0; i<10; i++))
    do
        /opt/jupyterhub/bin/python3 ../examples/ns_variable_size.py $dimension $i &
    done
done
wait
