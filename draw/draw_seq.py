import matplotlib.pyplot as plt
import numpy as np
import os


def cal_avg(folder, node_num):
    filenames = [os.path.join(folder, f"r{i}-tacc-poison-100.log") for i in range(0, node_num)]

    all_epochs = []

    for filename in filenames:
        with open(filename, 'r') as file:
            values = [float(line.strip()) for line in file if line.strip()]
            all_epochs.append(values)

    averages = [sum(epoch_values) / node_num for epoch_values in zip(*all_epochs)]

    return averages


def draw_nodes(folder, nodes):
    filenames = {node: f"{folder}r{node}-tacc-poison-100.log" for node in nodes}

    data = {}

    for node, filename in filenames.items():
        with open(filename, 'r') as file:
            data[f"node{node}"] = [float(line.strip()) for line in file if line.strip()]

    plt.figure(figsize=(9, 6))
    c = ["#BC96E6", "#086788", "#F0C808", "#DD1C1A","#483C46"]
    ind=0
    for node, values in data.items():
        plt.plot(values[:100], color=c[ind], label=node)
        ind+=1

    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Attach Success Rate',  fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(linestyle='--')
    plt.savefig("seq.pdf", dpi=150)
    plt.legend()
    plt.show()

nodes = [0, 1, 3, 5, 7]
draw_nodes("../Output/DBA-linear-swift-iid-DBA-datasets-linear8-all-test-200/adv3-all/", nodes)