#!/bin/bash

for shift in {6..12}
do
    mpirun --oversubscribe -np 10 python Train.py --config Config/trigger_cifar/location_effects/swift-iid-DBA-cifar-ring-trigger-location$shift.yaml --randomSeed 200
done

for size in {1..12}
do
    mpirun --oversubscribe -np 10 python Train.py --config Config/trigger_cifar/size_effects/swift-iid-DBA-cifar-ring-trigger-size$size.yaml --randomSeed 200
done