#!/bin/bash

NUM_ITER=50
for i in $(seq 1 $NUM_ITER)
do
    CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 python tdmpc2/collect.py task=walker_walk sb3_algo=sac steps=200000 morphology=True morphology_seed=$i ckpt_step=50000
done