#!/bin/bash

for seed in {0..100..10}
do
    mpirun --oversubscribe -np 10 python Train.py --config Config/dsgd-iid-DBA-1-global-200-10nodes-dataset.yaml --randomSeed $seed
    mpirun --oversubscribe -np 10 python Train.py --config Config/swift-iid-DBA-ring-1-global-200-10nodes-dataset.yaml --randomSeed $see
done
