#!/bin/bash

for clients_per_round in 2 4 8
do
    for num_epochs in 1 3 6 9 12
    do
        echo "--------------------------------Running the file with " $clients_per_round " clients and " $num_epochs " epochs-----------------------------------------" 
        python python test.py --task 2 --setting federated --dataset idda --dataset2 idda --model deeplabv3_mobilenetv2 --num_rounds 100 --num_epochs $num_epochs --clients_per_round $clients_per_round
    done
done