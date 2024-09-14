#!/bin/bash

for shift in {0..12}
do
    mpirun --oversubscribe -np 10 python Train.py --config Config/trigger_cifar/location_effects/swift-iid-DBA-cifar-ring-trigger-location$shift.yaml --randomSeed 100
done
