#!/bin/bash
# sudo apt-get install openmpi-bin openmpi-common libopenmpi-dev
# python 3.8

pip install torch==2.2.0+cu121 torchvision==0.17.0+cu121 -f https://download.pytorch.org/whl/torch_stable.html

pip install pyyaml numpy==1.19.5 networkx==2.6.3 Pillow==8.4.0

conda install -c conda-forge openmpi

conda install -c conda-forge mpi4py