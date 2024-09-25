#!/bin/bash

mpirun --oversubscribe -np 10 python Train.py --config Config/swift-iid-DBA-mnist-ring-0-benign.yaml --randomSeed 200
# mpirun --oversubscribe -np 10 python Train.py --config Config/swift-iid-DBA-mnist-ring-local-1.yaml --randomSeed 200
# mpirun --oversubscribe -np 10 python Train.py --config Config/swift-iid-DBA-mnist-ring-local-2.yaml --randomSeed 200
# mpirun --oversubscribe -np 10 python Train.py --config Config/swift-iid-DBA-mnist-ring-local-3.yaml --randomSeed 200