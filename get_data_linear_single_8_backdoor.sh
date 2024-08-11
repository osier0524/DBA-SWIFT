#!/bin/bash

for alt in {0..3..3}
do
    for seed in {10..105..5}
    do
        mpirun -np 8 python Train.py --config Config/swift-iid-linear-single-attack-8-backdoor-alt$alt.yaml --randomSeed $seed
    done
done