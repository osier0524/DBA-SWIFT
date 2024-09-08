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
    c=[ "#DD1C1A","#04E824", "#483C46"]
    marker = 'D'
    for i, folder in enumerate(folders):
        averages = cal_avg(folder, node_num)
        sample_avgs = averages[:15]
        epochs = list(range(1, 3001, 200))
        plt.plot(epochs, sample_avgs, marker=marker,color=c[i], label=legends[i])
    plt.ylim(0, 100)
    plt.xlabel('Epoch', fontsize=16)
    plt.ylabel('Attach Success Rate',  fontsize=16)
    plt.legend(fontsize=15,loc="lower right")
    plt.grid(linestyle='--')
    plt.savefig("res_DSGD_cliquering_cifar.pdf", dpi=150)
    plt.show()

# folders = ['../Output/DBA-ring-swift-iid-DBA-4-3001/adv3-all/',
#            '../Output/DBA-ring-swift-iid-DBA-4-1g3l-3000/adv3-all/',
#            '../Output/DBA-ring-swift-iid-DBA-4-1g1g-3000-old/adv3-all/']
# legends = ['DBA', 'Cluster-based DBA', 'Centralized in cluster']
# draw_together(folders, 40, legends)

# folders = ['../Output/DBA-clique-ring-swift-iid-DBA-4-3001/adv3-all/',
#            '../Output/DBA-clique-ring-swift-iid-DBA-4-1g3l-3000/adv3-all/',
#            '../Output/DBA-clique-ring-swift-iid-DBA-4-1g1g-3000/adv3-all/']

#
# folders = ['../Output/DBA-ring-dsgd-iid-DBA-4-3000/adv3-all/',
#            '../Output/DBA-ring-dsgd-iid-DBA-4-1g3l-3000/adv3-all/',
#            '../Output/DBA-ring-dsgd-iid-DBA-4-1g1g-3000/adv3-all/']

folders = ['../Output/DBA-clique-ring-dsgd-iid-DBA-4-3000/adv3-all/',
           '../Output/DBA-clique-ring-dsgd-iid-DBA-4-1g3l-3000/adv3-all/',
           '../Output/DBA-clique-ring-dsgd-iid-DBA-4-1g1g-3000/adv3-all/']

legends = ['DBA', 'Cluster-based DBA', 'Centralized attack in cluster']
draw_together(folders, 40, legends)