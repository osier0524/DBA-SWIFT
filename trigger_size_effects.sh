#!/bin/bash

for size in {11..12}
do
    mpirun --oversubscribe -np 10 python Train.py --config Config/trigger_cifar/size_effects/swift-iid-DBA-cifar-ring-trigger-size$size.yaml --randomSeed 200
done

for shift in {0..5}
do
    mpirun --oversubscribe -np 10 python Train.py --config Config/trigger_cifar/location_effects/swift-iid-DBA-cifar-ring-trigger-location$shift.yaml --randomSeed 200
done