#!/bin/bash

# bash halfcheetah_v4_sac/hparam_search.sh

epoch=10
eval_freq=4688

batch_sizes=(512 1024 2048)
lrs=(5e-3 1e-3 5e-4 1e-4)
bins=(31)

for bs in "${batch_sizes[@]}"
do
    for lr in "${lrs[@]}"
    do
        for b in "${bins[@]}"
        do
            echo "Running: bs=$bs, lr=$lr, bins=$b"

            logdir="halfcheetah_v4_sac/runs/hparam/bs${bs}_lr${lr}_b${b}"
            mkdir -p "$logdir"

            python halfcheetah_v4_sac/experiment.py \
                --batch_size $bs \
                --lr $lr \
                --eval_freq $eval_freq \
                --epochs $epoch \
                --bins $b \
                --logdir "$logdir"
        done
    done
done