#!/bin/bash

for seed in {0..100..10}
do
    # swift linear 10
    mpirun --oversubscribe -np 10 python Train.py --config Config/swift-iid-DBA-linear-1-global-200-10nodes.yaml --randomSeed $seed
done
