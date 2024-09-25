import numpy as np
import time
import argparse

import yaml
from GDM import Resnet
from GDM import MnistNet
from GDM.resnet_tinyimagenet import resnet18
from GDM.GraphConstruct import GraphConstruct
from Communicators.AsyncCommunicator import AsyncDecentralized
from Communicators.DSGD import decenCommunicator
from mpi4py import MPI
from GDM.DataPartition import partition_dataset
from Communicators.CommHelpers import flatten_tensors
from Utils.Misc import AverageMeter, Recorder, test_accuracy, test_accuracy_poison, test_loss, compute_accuracy
import os
import torch
import torch.utils.data.distributed
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import Poison
cudnn.benchmark = True

class ConfigObject(object):
    def __init__(self, dictionary):
        for key, value in dictionary.items():
            setattr(self, key, value)

def run(rank, size):

    # set random seed
    torch.manual_seed(args.randomSeed + rank)
    np.random.seed(args.randomSeed)

    # select neural network model
    num_class = 10
    if args.dataset == 'cifar10':
        model = Resnet.ResNet(args.resSize, num_class)
    elif args.dataset == 'mnist':
        model = MnistNet.MnistNet()
    elif args.dataset == 'tinyimagenet':
        model = resnet18(name='Local')

    # split up GPUs
    num_gpus = torch.cuda.device_count()
    gpu_id = rank % num_gpus

    # initialize the GPU being used
    torch.cuda.set_device(gpu_id)
    model = model.cuda(gpu_id)

    # model loss and optimizer
    criterion = nn.CrossEntropyLoss().cuda(gpu_id)
    optimizer = optim.SGD(model.parameters(),
                          lr=args.lr,
                          momentum=args.momentum,
                          weight_decay=1e-4,
                          nesterov=args.nesterov)

    # guarantee all local models start from the same point
    init_model = sync_allreduce(model, size, MPI.COMM_WORLD)

    # load data CIFAR10
    val_split = 0
    train_loader, test_loader, poison_test_loaders = partition_dataset(rank, size, MPI.COMM_WORLD, val_split, args)

    # ensure swift uses its own weighting
    if args.comm_style == 'swift':
        args.weight_type = 'swift'

    # load base network topology
    p = 3/size
    GP = GraphConstruct(rank, size, MPI.COMM_WORLD, args.graph, args.weight_type, p=p, num_c=args.num_clusters)

    if args.comm_style == 'swift':
        communicator = AsyncDecentralized(rank, size, MPI.COMM_WORLD, GP,
                                          args.sgd_steps, args.max_sgd, args.wb, args.memory_efficient, init_model)
    elif args.comm_style == 'ld-sgd':
        communicator = decenCommunicator(rank, size, MPI.COMM_WORLD, GP, args.i1, args.i2)
    elif args.comm_style == 'pd-sgd':
        communicator = decenCommunicator(rank, size, MPI.COMM_WORLD, GP, args.i1, 1)
    elif args.comm_style == 'd-sgd':
        communicator = decenCommunicator(rank, size, MPI.COMM_WORLD, GP, 0, 1)
    else:
        # Anything else just default to our algorithm
        communicator = AsyncDecentralized(rank, size, MPI.COMM_WORLD, GP,
                                          args.sgd_steps, args.max_sgd, args.wb, args.memory_efficient, init_model)

    # init recorder
    comp_time = 0
    comm_time = 0
    recorder = Recorder(args, rank)
    losses = AverageMeter()
    top1 = AverageMeter()

    for i in range(size):
        if rank == i:
            recorder.save_neighbors(communicator.neighbor_list)
        MPI.COMM_WORLD.Barrier()

    if args.noniid:
        d_epoch = 200
    else:
        d_epoch = 100

    # TODO: 根据rank来分配adversaries
    # adv_list = [17, 33, 77, 11]
    # adv_epoch = [[203], [205], [207], [209]]
    # poison_patterns = [[[0, 0], [0, 1], [0, 2], [0, 3], [0, 4], [0, 5]], 
    #                    [[0, 9], [0, 10], [0, 11], [0, 12], [0, 13], [0, 14]], 
    #                    [[4, 0], [4, 1], [4, 2], [4, 3], [4, 4], [4, 5]], 
    #                    [[4, 9], [4, 10], [4, 11], [4, 12], [4, 13], [4, 14]]]
    # trigger_num = 4
    # poison_label_swap = 2
    # poisoning_per_batch = 5

    poison = Poison.Poison(args)

    MPI.COMM_WORLD.Barrier()
    # start training
    for epoch in range(args.epoch):
        init_time = time.time()
        record_time = 0
        model.train()
        
        # Start training each epoch
        for batch_idx, (data, target) in enumerate(train_loader):
            start_time = time.time()
            # TODO: if rank is adversary, then poison the data
            if is_adversary(rank, epoch, args.adv_list, args.adv_epoch):
                if batch_idx == 0:
                    print(f"rank {rank} starts to poison")
                if args.adv_method == 'DBA':
                    data, target, poison_count = poison.get_poison_batch_DBA(data, target, args.adv_list[0], False)
                else:
                    data, target, poison_count = poison.get_poison_batch(data, target, rank, False)
            # TODO: if rank is benign, then use original code

            # data loading
            data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)

            # forward pass
            output = model(data)
            loss = criterion(output, target)

            # record training loss and accuracy
            record_start = time.time()
            acc1 = compute_accuracy(output, target)
            losses.update(loss.item(), data.size(0))
            top1.update(acc1[0].item(), data.size(0))
            record_end = time.time() - record_start
            record_time += record_end

            # backward pass
            loss.backward()

            # communication happens here
            comm_start = time.time()
            d_comm_time = communicator.communicate(model)
            comm_t = time.time() - comm_start

            # gradient step
            optimizer.step()
            optimizer.zero_grad()
            end_time = time.time()

            # compute computational time
            comp_time += (end_time - start_time - comm_t)

            # compute communication time
            comm_time += d_comm_time

        # update learning rate here
        if not args.customLR:
            update_learning_rate(optimizer, epoch, drop=0.5, epochs_drop=10.0, decay_epoch=d_epoch,
                                    itr_per_epoch=len(train_loader))
        else:
            if epoch == 81 or epoch == 122:
                args.lr *= 0.1
                for param_group in optimizer.param_groups:
                    param_group["lr"] = args.lr

        # evaluate test accuracy at the end of each epoch
        t = time.time()
        t_loss = test_loss(model, test_loader, criterion)
        test_time = time.time() - t

        # evaluate accuracy on poison test data
        # asrs = []
        poison_accs = []
        distributions = []
        if args.adv_mode == 'all2000' or args.adv_mode == 'zero2000':
            if epoch % 200 == 0:
                for i, poison_test_loader in enumerate(poison_test_loaders):
                    poison_acc, distribution = test_accuracy_poison(model, poison, args.poison_label_swap[i], poison_test_loader, adv_index=args.adv_list[0])
                    # asrs.append(asr)
                    poison_accs.append(poison_acc)
                    distributions.append(distribution)
        elif args.adv_mode == 'all-mnist':
            if epoch % 5 == 0:
                for i, poison_test_loader in enumerate(poison_test_loaders):
                    poison_acc, distribution = test_accuracy_poison(model, poison, args.poison_label_swap[i], poison_test_loader, adv_index=args.adv_list[0])
                    # asrs.append(asr)
                    poison_accs.append(poison_acc)
                    distributions.append(distribution)
        else:
            for i, poison_test_loader in enumerate(poison_test_loaders):
                poison_acc, distribution = test_accuracy_poison(model, poison, args.poison_label_swap[i], poison_test_loader, adv_index=args.adv_list[0])
                # asrs.append(asr)
                poison_accs.append(poison_acc)
                distributions.append(distribution)
        
        # evaluate validation accuracy at the end of each epoch
        # val_acc = test_accuracy(model, val_loader)

        # run personalization if turned on
        # if args.personalize and args.comm_style == 'swift':
        #    comm_time += communicator.personalize(epoch+2, val_acc, args.noniid)

        # total time spent in algorithm
        comp_time -= record_time
        epoch_time = comp_time + comm_time

        print("rank: %d, epoch: %.3f, loss: %.3f, train_acc: %.3f, test_loss: %.3f, comp time: %.3f, "
              "epoch time: %.3f" % (rank, epoch, losses.avg, top1.avg, t_loss, comp_time, epoch_time))
        print('poison accs: [{}]'.format(', '.join([str(x) for x in poison_accs])))
        # print('asr: [{}]'.format(', '.join([str(x) for x in asrs])))
        recorder.add_new(comp_time, comm_time, epoch_time, (time.time() - init_time)-test_time,
                         top1.avg, poison_accs, distributions, losses.avg, t_loss)
        # reset recorders
        comp_time, comm_time = 0, 0
        losses.reset()
        top1.reset()

        # recorder.save_parameters(model, epoch)

    # Save model to output folder
    # if rank == 0 and args.dataset == 'mnist':
    #     # models_path = './models' + '/' + args.name + '_rank0.pth'
    #     # torch.save(model.state_dict(), models_path)
    #     recorder.save_parameters(model, args.epoch)
    
    # Save data to output folder
    recorder.save_to_file()

    # Broadcast/wait until all other neighbors are finished in async algorithm
    if args.comm_style == 'swift' and args.memory_efficient:
        communicator.wait(model)
        print('Finished from Rank %d' % rank)

    MPI.COMM_WORLD.Barrier()

    sync_allreduce(model, size, MPI.COMM_WORLD)
    test_acc = test_accuracy(model, test_loader)
    print("rank %d: Test Accuracy %.3f" % (rank, test_acc))
    # if rank == 5:
    #     torch.save(model.state_dict(), args.outputFolder + '/models/' + args.name + '_rank5.pth')
    all_test_acc =MPI.COMM_WORLD.gather(test_acc, root=0)
    if rank == 0:
        saveFolderName = args.outputFolder + '/' + 'DBA-' + str(args.graph) + '-' + args.name + '-' + str(args.randomSeed)
        test_acc_file = os.path.join(saveFolderName, 'test_accuracies.txt')
        with open(test_acc_file, 'w') as f:
            for acc in all_test_acc:
                f.write("%.3f\n" % (acc))

# adv_list = [17, 33, 77, 11]
# adv_epoch = [[203], [205], [207], [209]]
def is_adversary(rank, epoch, adv_list, adv_epoch):
    if rank in adv_list:
        if args.adv_mode == 'regular':
            for i in range(len(adv_list)):
                if rank == adv_list[i] and epoch in adv_epoch[i]:
                    return True
        elif args.adv_mode == 'all':
            return True
        elif args.adv_mode == 'all-mnist':
            if epoch >= 25:
                return True
        elif args.adv_mode == 'all2000':
            if epoch >= 100:
                return True
        elif args.adv_mode == 'alter':
            if epoch % args.attack_interval == 0:
                return True
        elif args.adv_mode == 'zero2000':
            return False
    return False

def update_learning_rate(optimizer, epoch, drop, epochs_drop, decay_epoch, itr=None, itr_per_epoch=None):
    """
    1) Linearly warmup to reference learning rate (5 epochs)
    2) Decay learning rate exponentially starting at decay_epoch
    ** note: args.lr is the reference learning rate from which to scale up
    ** note: minimum global batch-size is 256
    """
    base_lr = 0.1
    lr = args.lr

    if args.warmup and epoch < 5:  # warmup to scaled lr
        if lr > base_lr:
            assert itr is not None and itr_per_epoch is not None
            count = epoch * itr_per_epoch + itr + 1
            incr = (lr - base_lr) * (count / (5 * itr_per_epoch))
            lr = base_lr + incr
    elif epoch >= decay_epoch:
        lr *= np.power(drop, np.floor((1 + epoch - decay_epoch) / epochs_drop))

    if lr is not None:
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


def sync_allreduce(model, size, comm):
    senddata = {}
    recvdata = {}
    for param in model.parameters():
        tmp = param.data.cpu()
        senddata[param] = tmp.numpy()
        recvdata[param] = np.empty(senddata[param].shape, dtype=senddata[param].dtype)
    torch.cuda.synchronize()
    comm.Barrier()

    for param in model.parameters():
        comm.Allreduce(senddata[param], recvdata[param], op=MPI.SUM)
    torch.cuda.synchronize()
    comm.Barrier()

    tensor_list = list()
    for param in model.parameters():
        tensor_list.append(param)
        param.data = torch.Tensor(recvdata[param]).cuda()
        param.data = param.data / float(size)

    # flatten tensors
    initial_model = flatten_tensors(tensor_list).cpu().detach().numpy()

    return initial_model


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--config', type=str, help='Path to the configuration file')
    parser.add_argument('--name', type=str, help='test case name')
    parser.add_argument('--randomSeed', type=int, help='random seed')
    parser.add_argument('--outputFolder', type=str, help='output folder')
    cmd_args = parser.parse_args()

    with open(cmd_args.config, 'r') as f:
        config = yaml.safe_load(f)
    args = ConfigObject(config)

    if cmd_args.name is not None:
        args.name = cmd_args.name
    if cmd_args.randomSeed is not None:
        args.randomSeed = cmd_args.randomSeed
    if cmd_args.outputFolder is not None:
        args.outputFolder = cmd_args.outputFolder

    if not args.description:
        print('Please input an experiment description. Exiting!')
        exit()

    if not os.path.isdir(args.outputFolder):
        os.mkdir(args.outputFolder)

    rank = MPI.COMM_WORLD.Get_rank()
    size = MPI.COMM_WORLD.Get_size()

    run(rank, size)
