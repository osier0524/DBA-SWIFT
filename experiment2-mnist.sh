#!/bin/bash


# swift mnist ring
mpirun --oversubscribe -np 40 python Train.py --config Config/swift-iid-DBA-mnist-ring-4.yaml --randomSeed 100
mpirun --oversubscribe -np 40 python Train.py --config Config/swift-iid-DBA-mnist-ring-4-1g1g.yaml --randomSeed 100
mpirun --oversubscribe -np 40 python Train.py --config Config/swift-iid-DBA-mnist-ring-4-1g3l.yaml --randomSeed 100

# swift mnist clique ring
mpirun --oversubscribe -np 40 python Train.py --config Config/swift-iid-DBA-mnist-clique-4.yaml --randomSeed 100
mpirun --oversubscribe -np 40 python Train.py --config Config/swift-iid-DBA-mnist-clique-4-1g1g.yaml --randomSeed 100
mpirun --oversubscribe -np 40 python Train.py --config Config/swift-iid-DBA-mnist-clique-4-1g3l.yaml --randomSeed 100


# dsgd mnist ring
mpirun --oversubscribe -np 40 python Train.py --config Config/dsgd-iid-DBA-mnist-ring-4.yaml --randomSeed 100
mpirun --oversubscribe -np 40 python Train.py --config Config/dsgd-iid-DBA-mnist-ring-4-1g1g.yaml --randomSeed 100
mpirun --oversubscribe -np 40 python Train.py --config Config/dsgd-iid-DBA-mnist-ring-4-1g3l.yaml --randomSeed 100

# dsgd mnist clique ring
mpirun --oversubscribe -np 40 python Train.py --config Config/dsgd-iid-DBA-mnist-clique-4.yaml --randomSeed 100
mpirun --oversubscribe -np 40 python Train.py --config Config/dsgd-iid-DBA-mnist-clique-4-1g1g.yaml --randomSeed 100
mpirun --oversubscribe -np 40 python Train.py --config Config/dsgd-iid-DBA-mnist-clique-4-1g3l.yaml --randomSeed 100
