#!/bin/bash

for seed in {10..105..5}
do
    mpirun -np 8 python Train.py --config Config/swift-iid-linear-multi-attack-8.yaml --randomSeed $seed
done
