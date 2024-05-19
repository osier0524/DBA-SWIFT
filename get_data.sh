#!/bin/bash

for seed in {0..100..10}
do
    mpirun -np 8 python Train.py --config Config/swift-iid-linear-shark.yaml --randomSeed $seed
done
