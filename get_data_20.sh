#!/bin/bash

for seed in {5..105..10}
do
    # dsgd linear 20
    mpirun --oversubscribe -np 20 python Train.py --config Config/dsgd-iid-DBA-linear-1-global-200-20nodes-dataset.yaml --randomSeed $seed
done
