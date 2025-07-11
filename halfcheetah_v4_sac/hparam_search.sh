#!/bin/bash

# bash halfcheetah_v4_sac/hparam_search.sh

# so far, it's bs=512, lr=1e-3
# bin=6 seems the best, 8 doesnt seem bad either

epoch=5
eval_freq=4688

batch_sizes=(512 1024 2048)
lrs=(5e-3 1e-3 5e-4 1e-4)
# bins=(31)

# batch_sizes=(512)
# lrs=(1e-3)
# bins=(11 25 31 41 51)
# bins=(5 6 7 8 9 10)

for bs in "${batch_sizes[@]}"
do
    for lr in "${lrs[@]}"
    do
        # for b in "${bins[@]}"
        # do
        echo "Running: bs=$bs, lr=$lr"

        logdir="halfcheetah_v4_sac/runs/logprob/bs${bs}_lr${lr}"
        mkdir -p "$logdir"

        python halfcheetah_v4_sac/experiment.py \
            --batch_size $bs \
            --lr $lr \
            --eval_freq $eval_freq \
            --epochs $epoch \
            --logdir "$logdir"
        # done
    done
done