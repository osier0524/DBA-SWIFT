#!/bin/bash

for seed in {210..230..10}
do
    # dsgd ring 20
    mpirun --oversubscribe -np 20 python Train.py --config Config/dsgd-iid-DBA-1-global-200-20nodes-dataset.yaml --randomSeed $seed
done

for seed in {210..230..10}
do
    # dsgd ring 10
    mpirun --oversubscribe -np 10 python Train.py --config Config/dsgd-iid-DBA-1-global-200-10nodes-dataset.yaml --randomSeed $seed
done