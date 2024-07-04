#!/bin/bash

for seed in {0..100..10}
do
    mpirun --oversubscribe -np 40 python Train.py --config Config/swift-iid-linear-multi-attack.yaml --randomSeed $seed
done
