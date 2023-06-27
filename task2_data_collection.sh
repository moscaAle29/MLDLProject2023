#!/bin/bash

for clients_per_round in 1 2 3 4 5 6 7 8
do
    for num_epochs in 1 3 6 9
    do
        for num_rounds in 50 100 200 
        do
            python3 main.py --setting centralized --dataset gta5 --dataset2 idda --model deeplabv3_mobilenetv2 --num_rounds $num_rounds --num_epochs $num_epochs --clients_per_round $clients_per_round --lr 0.01 --bs 8 --rrc_transform --jitter --domain_adapt fda --fda_alpha 0.01 --load_pretrained --round 30 --resume --run_path flproject2023/centralized_gta5_idda/un91c1w7
        done
    done
done