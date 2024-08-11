#!/bin/bash

for seed in {0..100..10}
do
    mpirun --oversubscribe -np 20 python Train.py --config Config/swift-iid-DBA-ring-1-global-200-20nodes-dataset.yaml --randomSeed $seed
done
