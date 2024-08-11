#!/bin/bash

for seed in {10..50..10}
do
    mpirun --oversubscribe -np 40 python Train.py --config Config/swift-iid-ring-multi-attack-40.yaml --randomSeed $seed
done
