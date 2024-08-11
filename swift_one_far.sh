#!/bin/bash



mpirun --oversubscribe -np 40 python Train.py --config Config/swift-iid-DBA-4-far-global-clustered-global.yaml --randomSeed 100
mpirun --oversubscribe -np 40 python Train.py --config Config/swift-iid-DBA-1-cluster-local.yaml --randomSeed 100
mpirun --oversubscribe -np 40 python Train.py --config Config/swift-iid-DBA-1-global.yaml --randomSeed 100