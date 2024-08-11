#!/bin/bash

for num in {0..6}
do
    mpirun --oversubscribe -np 40 python Train.py --config Config/swift-iid-DBA-$num.yaml --randomSeed 100
done

for num in {0..6}
do
    mpirun --oversubscribe -np 40 python Train.py --config Config/swift-iid-DBA-clique-$num.yaml --randomSeed 100
done