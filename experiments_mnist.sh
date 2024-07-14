#!/bin/bash

for num in {1..6}
do
    mpirun --oversubscribe -np 40 python Train.py --config Config/swift-iid-DBA-mnist-ring-$num.yaml --randomSeed 100
done
