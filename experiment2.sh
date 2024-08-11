#!/bin/bash


# swift ring exp3
mpirun --oversubscribe -np 40 python Train.py --config Config/swift-iid-DBA-4-far-global-clustered-global.yaml --randomSeed 100

# swift clique ring
mpirun --oversubscribe -np 40 python Train.py --config Config/swift-iid-DBA-clique-4-far-global-clustered-global.yaml --randomSeed 100
mpirun --oversubscribe -np 40 python Train.py --config Config/swift-iid-DBA-clique-4-far-global-clustered-local.yaml --randomSeed 100


# dsgd ring
mpirun --oversubscribe -np 40 python Train.py --config Config/dsgd-iid-DBA-4.yaml --randomSeed 100
mpirun --oversubscribe -np 40 python Train.py --config Config/dsgd-iid-DBA-4-far-global-clustered-global.yaml --randomSeed 100
mpirun --oversubscribe -np 40 python Train.py --config Config/dsgd-iid-DBA-4-far-global-clustered-local.yaml --randomSeed 100


# dsgd clique ring
mpirun --oversubscribe -np 40 python Train.py --config Config/dsgd-iid-DBA-clique-4.yaml --randomSeed 100
mpirun --oversubscribe -np 40 python Train.py --config Config/dsgd-iid-DBA-clique-4-far-global-clustered-global.yaml --randomSeed 100
mpirun --oversubscribe -np 40 python Train.py --config Config/dsgd-iid-DBA-clique-4-far-global-clustered-local.yaml --randomSeed 100