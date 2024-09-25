#!/bin/bash

# for gap_size in {0..12}
# do
#     mpirun --oversubscribe -np 40 python Train.py --config Config/trigger_cifar/gap_effects/swift-iid-DBA-cifar-ring-trigger-gap$gap_size.yaml --randomSeed 200
# done

# mnist
for gap_size in {0..12}
do
    mpirun --oversubscribe -np 40 python Train.py --config Config/trigger_gap_effects/swift-iid-DBA-mnist-ring-trigger-gap$gap_size.yaml --randomSeed 200
done
