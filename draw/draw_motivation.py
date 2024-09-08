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

def draw_together(folders, node_num, legends):
    plt.figure(figsize=(9, 6))
    c=["#808080","#97CCE8","#60B177","#E18683"]
    marker = 'D'
    for i, folder in enumerate(folders):
        averages = cal_avg(folder, node_num)
        sample_avgs = averages[:15]
        epochs = list(range(1, 3001, 200))
        plt.plot(epochs, sample_avgs, marker=marker,color=c[i], label=legends[i])
    plt.ylim(0, 80)
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Attach Success Rate',  fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(linestyle='--')
    plt.savefig("motivation.pdf", dpi=150)
    plt.show()

folders = ['../Output/DBA-ring-swift-iid-DBA-0-3001/adv3-all/',
           '../Output/DBA-ring-swift-iid-DBA-1-1g-3000/adv3-all/',
           '../Output/DBA-ring-swift-iid-DBA-1-3001/adv3-all/',
           '../Output/DBA-ring-swift-iid-DBA-1-cluster-4l-3000/adv3-all/',]
legends = ['No Attack',
           'Centralized Attack',
           'Distributed Attack, Uniform Distribution',
           'Distributed Attack, Non-Uniform Distribution']
draw_together(folders, 40, legends)