#!/bin/bash

for seed in {210..230..10}
do
    # dsgd linear 20
    mpirun --oversubscribe -np 20 python Train.py --config Config/swift-iid-DBA-linear-1-global-200-20nodes.yaml --randomSeed $seed
done
