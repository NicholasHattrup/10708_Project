import os, sys
import argparse
import json
from datetime import datetime as dt
import time

import torch
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable

from sklearn.decomposition import PCA

# Our own modules
from learner import *
from models import MPNN

parser = argparse.ArgumentParser(description="Substructure graph with neural message passing")

parser.add_argument('--dataset', type=str, default='qm9',
                    help='aataset of interest, default: QM9')
parser.add_argument('--datasetPath', type=str, default='./data/qm9/xyz/',
                    help='dataset path, default: ./data/qm9/xyz/')
parser.add_argument('--datasetSplitDone', type=bool, default=True,
                    help='whether or not to use pre-split dataset, default: True')
parser.add_argument('--splitRatio', type=str, default='10000_10000',
                    help='split ratio, *validation_test*, with automated train')
parser.add_argument('--n-pcs', type=str, default='32_16',
                    help='number of principle components to use, *node_edge*, float would be for explained variance ratio (max 128_64)')
parser.add_argument('--pcPath', type=str, default='./data/qm9/xyz/',
                    help="path to pre-trained PCA model")
parser.add_argument('--logPath', type=str, default='./log_raw_distance_noHs/qm9/mpnn/', help='log path')
parser.add_argument('--plotLr', type=bool, default=False, help='allow plotting the data')
parser.add_argument('--plotPath', type=str, default='./plot/qm9/mpnn/', help='plot path')
parser.add_argument('--resume', type=str, default='./raw_distance_noHs/qm9/mpnn/',
                    help='path to latest checkpoint')
# Optimization Options
parser.add_argument('--batch-size', type=int, default=100, metavar='N',
                    help='Input batch size for training (default: 20)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Enables CUDA training')
parser.add_argument('--epochs', type=int, default=360, metavar='N',
                    help='Number of epochs to train (default: 360)')
parser.add_argument('--lr', type=float_bounds([1e-5, 1e-2]), default=1e-3, metavar='LR',
                    help='Initial learning rate [1e-5, 5e-4] (default: 1e-3)')
parser.add_argument('--lr-decay', type=float_bounds([.01, 1]), default=0.6, metavar='LR-DECAY',
                    help='Learning rate decay factor [.01, 1] (default: 0.6)')
parser.add_argument('--schedule', type=list, default=[0.1, 0.9], metavar='S',
                    help='Percentage of epochs to start the learning rate decay [0, 1] (default: [0.1, 0.9])')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')
# i/o
parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                    help='How many batches to wait before logging training status')
# Accelerating
parser.add_argument('--prefetch', type=int, default=2, help='Pre-fetching threads.')

parser.add_argument('--distance', type=bool, default=True,
                    help="whether of not use weighted distance as a feature for edges")

best_er1 = 0

def main():

    global args, best_er1
    args = parser.parse_args()
    start_epoch = 0
    
    # Check if CUDA is enabled
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    # Load data
    root = args.datasetPath
    
    files = [f for f in os.listdir(root) if f.endswith(".xyz")]
    split_path = "/".join(root.split('/')[:-2]) + '/'
    valid_ids, test_ids, train_ids = split_files(split_path=split_path, files=files, args=args)

    t0 = dt.now()
    train_lib = GraphLibrary(directory=root, filenames=train_ids[0:15000])
    valid_lib = GraphLibrary(directory=root, filenames=valid_ids[0:1])
    test_lib = GraphLibrary(directory=root, filenames=test_ids[0:1])
    print("Building libraries took: ", dt.now() - t0)

    KEY, libs = train_lib.MD5, [train_lib, valid_lib, test_lib]
    NodeConverter, EdgeConverter, DistanceConverter = GetCustomizedPCA(libs, args.n_pcs, KEY, modelPath=split_path)

    train_lib.update_library(NodeConverter, EdgeConverter, DistanceConverter)
    valid_lib.update_library(NodeConverter, EdgeConverter, DistanceConverter)
    test_lib.update_library(NodeConverter, EdgeConverter, DistanceConverter)

    train_loader = torch.utils.data.DataLoader(train_lib,
                                               batch_size=args.batch_size, shuffle=True,
                                               collate_fn=utils.collate,                                                                                                                                                                              
                                               num_workers=args.prefetch, pin_memory=True)

    valid_loader = torch.utils.data.DataLoader(valid_lib,
                                               batch_size=args.batch_size, shuffle=True,
                                               collate_fn=utils.collate,                                                                                                                                                                              
                                               num_workers=args.prefetch, pin_memory=True)

    test_loader = torch.utils.data.DataLoader(test_lib,
                                               batch_size=args.batch_size, shuffle=True,
                                               collate_fn=utils.collate,                                                                                                                                                                              
                                               num_workers=args.prefetch, pin_memory=True)

    g_tuple, target = train_lib[0]
    g, nodes, edges = g_tuple
    
    print('Creating Model',flush=True)
    in_n = [len(nodes[0]), len(list(edges.values())[0])]
    hidden_state_size = 73
    message_size = 73
    n_layers = 3
    l_target = len(target)
    type ='regression'
    model = MPNN(in_n, hidden_state_size, message_size, n_layers, l_target, type=type)
    del in_n, hidden_state_size, message_size, n_layers, l_target, type

    print('Creating optimizer',flush=True)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()
    evaluation = lambda output, target: torch.mean(torch.abs(output - target) / torch.abs(target))

    print('Creating logger',flush=True)
    logger = Logger(args.logPath)
    lr_step = (args.lr-args.lr*args.lr_decay)/(args.epochs*args.schedule[1] - args.epochs*args.schedule[0])

    # search for existing checkpoint
    if args.resume:
        look_for_best(args.resume, model, optimizer)
    
    # check for GPU
    print('check cuda',flush=True)
    if args.cuda:
        print('\t* cuda',flush=True)
        model = model.cuda()
        criterion = criterion.cuda()

    # run training epochs 
    for epoch in range(start_epoch, args.epochs):
        if epoch > args.epochs * args.schedule[0] and epoch < args.epochs * args.schedule[1]:
            args.lr -= lr_step
            for param_group in optimizer.param_groups:
                param_group['lr'] = args.lr

        # train for one epoch
        train(train_loader, model, args.cuda, criterion, optimizer, epoch, evaluation, int(args.log_interval), logger)

        # evaluate on test set                                                                                                                                                                                                                             
        er1, output, target = validate(valid_loader, model, criterion, evaluation, args.cuda, int(args.log_interval), logger)
        is_best = er1 > best_er1
        best_er1 = min(er1, best_er1)
        utils.save_checkpoint(s={'epoch': epoch + 1, 'state_dict': model.state_dict(), 'best_er1': best_er1,
                               'optimizer': optimizer.state_dict(), }, is_best=is_best, path=args.resume)

        # Logger step                                                                                                                                                                                                                                      
        logger.log_value('learning_rate', args.lr).step()

    # load best model
    look_for_best(args.resume, model, optimizer)

    # run validation on test set
    validate(test_loader, model, criterion, evaluation, args.cuda, int(args.log_interval))
            
if __name__ == '__main__':
    main()
