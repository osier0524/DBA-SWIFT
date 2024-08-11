#!/bin/bash

for seed in {35..105..10}
do
    mpirun -np 8 python Train.py --config Config/swift-iid-linear-single-attack-8-alt1.yaml --randomSeed $seed
done

for alt in {2..5}
do 
    for seed in {15..105..10}
    do
        mpirun -np 8 python Train.py --config Config/swift-iid-linear-single-attack-8-alt$alt.yaml --randomSeed $seed
    done
done