#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
    Trains a Neural Message Passing Model on various datasets. Methodologi defined in:

    Gilmer, J., Schoenholz S.S., Riley, P.F., Vinyals, O., Dahl, G.E. (2017)
    Neural Message Passing for Quantum Chemistry.
    arXiv preprint arXiv:1704.01212 [cs.LG]

"""

# Torch
import torch
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable

import time
import argparse
import os
import numpy as np

# Our Modules
import datasets
from datasets import utils
from models.MPNN import MPNN
from LogMetric import AverageMeter, Logger

__author__ = "Pau Riba, Anjan Dutta"
__email__ = "priba@cvc.uab.cat, adutta@cvc.uab.cat"
#lambda x
def abcd(x, stat_dict1):
    return datasets.utils.normalize_data(x,stat_dict1['target_mean'],stat_dict1['target_std'])

# Parser check
def restricted_float(x, inter):
    x = float(x)
    if x < inter[0] or x > inter[1]:
        raise argparse.ArgumentTypeError("%r not in range [1e-5, 1e-4]"%(x,))
    return x

# Argument parser
parser = argparse.ArgumentParser(description='Neural message passing')

parser.add_argument('--dataset', default='qm9', help='QM9')
parser.add_argument('--datasetPath', default='./data/qm9/dsgdb9nsd/', help='dataset path')
parser.add_argument('--logPath', default='./rdist_model_log/qm9/mpnn/', help='log path')
parser.add_argument('--plotLr', default=False, help='allow plotting the data')
parser.add_argument('--plotPath', default='./plot/qm9/mpnn/', help='plot path')
parser.add_argument('--resume', default='./rdist_model_checkpoint/qm9/mpnn/',
                    help='path to latest checkpoint')
# Optimization Options
parser.add_argument('--batch-size', type=int, default=10, metavar='N',
                    help='Input batch size for training (default: 20)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Enables CUDA training')
parser.add_argument('--epochs', type=int, default=360, metavar='N',
                    help='Number of epochs to train (default: 360)')
parser.add_argument('--lr', type=lambda x: restricted_float(x, [1e-5, 1e-2]), default=1e-3, metavar='LR',
                    help='Initial learning rate [1e-5, 5e-4] (default: 1e-4)')
parser.add_argument('--lr-decay', type=lambda x: restricted_float(x, [.01, 1]), default=0.6, metavar='LR-DECAY',
                    help='Learning rate decay factor [.01, 1] (default: 0.6)')
parser.add_argument('--schedule', type=list, default=[0.1, 0.9], metavar='S',
                    help='Percentage of epochs to start the learning rate decay [0, 1] (default: [0.1, 0.9])')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')
# i/o
parser.add_argument('--log-interval', type=int, default=20, metavar='N',
                    help='How many batches to wait before logging training status')
# Accelerating
parser.add_argument('--prefetch', type=int, default=2, help='Pre-fetching threads.')

parser.add_argument('--e_rep', type=str, default="chem_graph", help="edge representation")

best_er1 = 0

def main():

    global args, best_er1
    args = parser.parse_args()

    # Check if CUDA is enabled
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    # Load data
    root = args.datasetPath

    print('Prepare files',flush=True)

    if args.dataset == 'qm9':
        train_ids = []
        valid_ids = []
        test_ids = []
        with open(root+"train_ids.txt", "r") as f:
            for line in f:
                train_ids.append(line.replace("\n",""))

        with open(root+"valid_ids.txt", "r") as f:
            for line in f:
                train_ids.append(line.replace("\n",""))

        with open(root+"test_ids.txt", "r") as f:
            for line in f:
                train_ids.append(line.replace("\n",""))
    else:
        test_ids = [f for f in os.listdir(root) if os.path.isfile(os.path.join(root, f))]
    e_representation = args.e_rep
    data_train = datasets.Qm9(root, train_ids, edge_transform=utils.qm9_edges, e_representation=e_representation)
    data_valid = datasets.Qm9(root, valid_ids, edge_transform=utils.qm9_edges, e_representation=e_representation)
    data_test = datasets.Qm9(root, test_ids, edge_transform=utils.qm9_edges, e_representation=e_representation)

    # Define model and optimizer
    print('Define model',flush=True)
    # Select one graph
    g_tuple, l = data_train[0]
    g, h_t, e = g_tuple

    # Data Loader

    test_loader = torch.utils.data.DataLoader(data_test,
                                              batch_size=args.batch_size, collate_fn=datasets.utils.collate_g,
                                              num_workers=args.prefetch, pin_memory=True)

    print('\tCreate model',flush=True)
    in_n = [len(h_t[0]), len(list(e.values())[0])]
    hidden_state_size = 73
    message_size = 73
    n_layers = 3
    l_target = len(l)
    type ='regression'
    model = MPNN(in_n, hidden_state_size, message_size, n_layers, l_target, type=type)
    del in_n, hidden_state_size, message_size, n_layers, l_target, type

    print('Optimizer',flush=True)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    criterion = nn.MSELoss()

    evaluation = lambda output, target: torch.mean(torch.abs(output - target) / torch.abs(target))

    print('Logger',flush=True)
    logger = Logger(args.logPath)

    # get the best checkpoint if available without training
    if args.resume:
        checkpoint_dir = args.resume
        best_model_file = os.path.join(checkpoint_dir, 'model_best.pth')
        if not os.path.isdir(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        if os.path.isfile(best_model_file):
            print("=> loading best model '{}'".format(best_model_file),flush=True)
            checkpoint = torch.load(best_model_file)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_er1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded best model '{}' (epoch {})".format(best_model_file, checkpoint['epoch']),flush=True)
        else:
            print("=> no best model found at '{}'".format(best_model_file),flush=True)

    print('Check cuda',flush=True)
    if args.cuda:
        print('\t* Cuda',flush=True)
        model = model.cuda()
        criterion = criterion.cuda()

    with open("analysis.csv","w") as f:
        f.write("prediction,ground truth\n")
    validate(test_loader, model, criterion, evaluation)
    
def validate(val_loader, model, criterion, evaluation, logger=None):
    batch_time = AverageMeter()
    losses = AverageMeter()
    error_ratio = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (g, h, e, target) in enumerate(val_loader):

        # Prepare input data
        if args.cuda:
            g, h, e, target = g.cuda(), h.cuda(), e.cuda(), target.cuda()
        g, h, e, target = Variable(g), Variable(h), Variable(e), Variable(target)
        
        # Compute output
        if i % args.log_interval == 0 and i > 0:
            print(i)

        output = model(g, h, e)
        predictions = output.tolist()
        ground_truth = target.tolist()
        with open("analysis.csv", "a") as f:
            for i, val in enumerate(predictions):
                f.write("{},{}\n".format(round(val[0],4), round(ground_truth[i][0],4)))
        # Logs
        losses.update(criterion(output, target).data, g.size(0))
        error_ratio.update(evaluation(output, target).data, g.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    print(' * Average Error Ratio {err.avg:.3f}; Average Loss {loss.avg:.3f}'
          .format(err=error_ratio, loss=losses),flush=True)

    if logger is not None:
        logger.log_value('test_epoch_loss', losses.avg)
        logger.log_value('test_epoch_error_ratio', error_ratio.avg)

    return error_ratio.avg, output, target

    
if __name__ == '__main__':
    main()
