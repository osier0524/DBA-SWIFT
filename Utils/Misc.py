import torch
import numpy as np
import os

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Recorder(object):
    def __init__(self, args, rank):
        # self.record_valacc = list()
        self.record_timing = list()
        self.record_total_timing = list()
        self.record_comp_timing = list()
        self.record_comm_timing = list()
        self.record_losses = list()
        self.record_trainacc = list()
        self.record_poisonaccs = list()
        for _ in args.poison_label_swap:
            self.record_poisonaccs.append(list())
        self.record_distributions = list()
        self.record_testloss = list()
        self.total_record_timing = list()
        self.record_neighbor_list = list()
        self.args = args
        self.rank = rank
        self.saveFolderName = args.outputFolder + '/' + self.args.name + '-' + str(self.args.graph) + '-' \
                              + str(self.args.sgd_steps) + 'sgd-' + str(self.args.epoch) + 'epochs'
        # self.saveFolderName = '/home/bhan/bohan/DBA-SWIFT/'
        if rank == 0 and not os.path.isdir(self.saveFolderName):
            os.mkdir(self.saveFolderName)

    def add_new(self, comp_time, comm_time, epoch_time, total_time, top1, poison_accs, distributions, losses, test_loss):
        self.record_timing.append(epoch_time)
        self.record_total_timing.append(total_time)
        self.record_comp_timing.append(comp_time)
        self.record_comm_timing.append(comm_time)
        self.record_trainacc.append(top1)
        for i, acc in enumerate(poison_accs):
            self.record_poisonaccs[i].append(acc)
        self.record_distributions.append(distributions)
        self.record_losses.append(losses)
        # self.record_valacc.append(val_acc)
        self.record_testloss.append(test_loss)
    
    def save_neighbors(self, neighbor_list):
        for n in neighbor_list:
            self.record_neighbor_list.append(n)
        
        np.savetxt(self.saveFolderName + '/r' + str(self.rank) + '-neighbors.log', self.record_neighbor_list, delimiter=',')

    def save_parameters(self, model, epoch):
        torch.save(model.state_dict(), self.saveFolderName + '/r' + str(self.rank) + '-epoch_' + str(epoch) + '-parameters.pt')

    def save_to_file(self):
        # np.savetxt(self.saveFolderName + '/r' + str(self.rank) + '-epoch-time.log', self.record_timing, delimiter=',')
        # np.savetxt(self.saveFolderName + '/r' + str(self.rank) + '-total-time.log', self.record_total_timing,
        #            delimiter=',')
        # np.savetxt(self.saveFolderName + '/r' + str(self.rank) + '-comptime.log', self.record_comp_timing,
        #            delimiter=',')
        # np.savetxt(self.saveFolderName + '/r' + str(self.rank) + '-commtime.log', self.record_comm_timing,
        #            delimiter=',')
        # np.savetxt(self.saveFolderName + '/r' + str(self.rank) + '-losses.log', self.record_losses, delimiter=',')
        # np.savetxt(self.saveFolderName + '/r' + str(self.rank) + '-tacc.log', self.record_trainacc, delimiter=',')
        # # np.savetxt(self.saveFolderName + '/r' + str(self.rank) + '-vacc.log', self.record_valacc, delimiter=',')
        # np.savetxt(self.saveFolderName + '/r' + str(self.rank) + '-testloss.log', self.record_testloss, delimiter=',')
        for i, record_poisonacc in enumerate(self.record_poisonaccs):
            np.savetxt(self.saveFolderName + '/adv' + str(self.args.attack_interval) + '/' + self.args.poison_labels[i] + '/r' + str(self.rank) + '-tacc-poison-' + str(self.args.randomSeed) + '.log', record_poisonacc, delimiter=',')

        np.save(self.saveFolderName + '/adv_distributions' + str(self.args.attack_interval) + '/r' + str(self.rank) + '-distributions-' + str(self.args.randomSeed) + '.npy', np.array(self.record_distributions))
        
        # with open(self.saveFolderName + '/ExpDescription', 'w') as f:
        #     f.write(str(self.args) + '\n')
        #     f.write(self.args.description + '\n')


def compute_accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def test_accuracy(model, test_loader):
    model.eval()
    top1 = AverageMeter()
    for batch_idx, (inputs, targets) in enumerate(test_loader):
        inputs, targets = inputs.cuda(non_blocking=True), targets.cuda(non_blocking=True)
        # compute output
        with torch.no_grad():
            outputs = model(inputs)
        acc1 = compute_accuracy(outputs, targets)
        top1.update(acc1[0].item(), inputs.size(0))
    return top1.avg


def test_loss(model, test_loader, criterion):
    model.eval()
    top1 = AverageMeter()
    for batch_idx, (inputs, targets) in enumerate(test_loader):
        inputs, targets = inputs.cuda(non_blocking=True), targets.cuda(non_blocking=True)
        # compute output
        with torch.no_grad():
            outputs = model(inputs)
        loss = criterion(outputs, targets)
        top1.update(loss.item(), inputs.size(0))
    return top1.avg

# test accuracy with poison
def test_accuracy_poison(model, poison, poison_test_loader, adv_index=-1):
    model.eval()
    top1 = AverageMeter()
    output_distributions = []

    for batch_idx, (inputs, targets) in enumerate(poison_test_loader):
        if poison.adv_method == 'DBA':
            inputs, targets, poison_num = poison.get_poison_batch_DBA(inputs, targets, adversarial_index=adv_index, evaluation=True)
        else:
            inputs, targets, poison_num = poison.get_poison_batch(inputs, targets, adversarial_index=adv_index, evaluation=True)
        inputs, targets = inputs.cuda(non_blocking=True), targets.cuda(non_blocking=True)
        # compute output
        with torch.no_grad():
            outputs = model(inputs)
            probabilities = torch.softmax(outputs, dim=1)

        acc1 = compute_accuracy(outputs, targets)
        top1.update(acc1[0].item(), inputs.size(0))

        output_distributions.append(probabilities.cpu().numpy())
    
    aggregated_distributions = np.concatenate(output_distributions, axis=0)

    return top1.avg, aggregated_distributions
