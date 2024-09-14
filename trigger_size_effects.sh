#!/bin/bash

for size in {1..12}
do
    mpirun --oversubscribe -np 10 python Train.py --config Config/trigger_cifar/size_effects/swift-iid-DBA-cifar-ring-trigger-size$size.yaml --randomSeed 100
done
