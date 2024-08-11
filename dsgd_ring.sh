#!/bin/bash



mpirun --oversubscribe -np 40 python Train.py --config Config/dsgd-iid-DBA-6.yaml --randomSeed 100
for num in {1..6}
do
    mpirun --oversubscribe -np 40 python Train.py --config Config/dsgd-iid-DBA-clique-$num.yaml --randomSeed 100
done