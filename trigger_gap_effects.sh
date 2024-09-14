#!/bin/bash

for gap_size in {0..12}
do
    mpirun --oversubscribe -np 10 python Train.py --config Config/trigger_cifar/gap_effects/swift-iid-DBA-cifar-ring-trigger-gap$gap_size.yaml --randomSeed 100
done
