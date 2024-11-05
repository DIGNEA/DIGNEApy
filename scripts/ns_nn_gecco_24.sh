#!/bin/bash


for ((i=0; i<10; i++))
do
        echo "Running $i"
        /opt/jupyterhub/bin/python3 gecco_24_gen_best_nn.py $i &
done
wait
#done
