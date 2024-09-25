#!/bin/bash

for shift in {0..12}
do
    mpirun --oversubscribe -np 40 python Train.py --config Config/trigger_location_effects/swift-iid-DBA-mnist-ring-trigger-location$shift.yaml --randomSeed 200
done