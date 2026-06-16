#!/bin/bash
benchmark="warm_start"

for seed in {1..3}; do
    python train_st.py --seed "$seed" --benchmark "$benchmark" --model RESNET18 --task CIFAR10
done
