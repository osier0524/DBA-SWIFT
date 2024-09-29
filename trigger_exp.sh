#!/bin/bash

# trigger gap
for gap_size in {0..12}
do
    mpirun --oversubscribe -np 40 python Train.py --config Config/trigger_cifar/gap_effects/swift-iid-DBA-cifar-ring-trigger-gap$gap_size.yaml --randomSeed 200
done

# tigger location
for shift in {0..12}
do
    mpirun --oversubscribe -np 40 python Train.py --config Config/trigger_cifar/location_effects/swift-iid-DBA-cifar-ring-trigger-location$shift.yaml --randomSeed 200
done

# trigger size
for size in {1..12}
do
    mpirun --oversubscribe -np 40 python Train.py --config Config/trigger_cifar/size_effects/swift-iid-DBA-cifar-ring-trigger-size$size.yaml --randomSeed 200
done
