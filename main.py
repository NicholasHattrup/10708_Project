import os, sys
import argparse
import json
from datetime import datetime as dt

import torch
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable

from sklearn.decomposition import PCA

# Our own modules
import utils
from utils import *
from ChemGraph import *

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


def main():

    args = parser.parse_args()
    start_epoch = 0
    
    # Check if CUDA is enabled
    # args.cuda = not args.no_cuda and torch.cuda.is_available()

    # Load data
    root = args.datasetPath
    
    files = [f for f in os.listdir(root) if f.endswith(".xyz")]
    split_path = "/".join(root.split('/')[:-2]) + '/'
    valid_ids, test_ids, train_ids = split_files(split_path=split_path, files=files, args=args)

    t0 = dt.now()
    train_lib = GraphLibrary(directory=root, filenames=train_ids)
    valid_lib = GraphLibrary(directory=root, filenames=valid_ids)
    test_lib = GraphLibrary(directory=root, filenames=test_ids)
    print("Building libraries took: ", dt.now() - t0)

    t0 = dt.now()
    KEY = train_lib.MD5
    libs = [train_lib, valid_lib, test_lib]
    NodeConverter, EdgeConverter, DistanceConverter = GetCustomizedPCA(libs, args.n_pcs, KEY, modelPath=split_path)
    print("Establishing PCA took", dt.now() - t0)

    t1 = dt.now()
    train_lib.update_library(NodeConverter, EdgeConverter)
    valid_lib.update_library(NodeConverter, EdgeConverter)
    test_lib.update_library(NodeConverter, EdgeConverter)
    print("Updating libraries took", dt.now()-t1)

    print(f"Congrats! PCA is done!")
    sys.exit(0)

    graph_lib = GraphLibrary(directory=root, filenames=train_ids[:100])
    graph_lib.init_reduction()
    fragments = graph_lib.fragment_library
    linkages = graph_lib.linkage_library

    t3 = dt.now()
    frag_fps = [Chem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(smi),2,nBits=1024) for smi in fragments]
    link_fps = [Chem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(smi),2,nBits=1024) for smi in linkages]

    nc = 50
    pca_frag = PCA().fit(frag_fps)
    ratio_frag = pca_frag.explained_variance_ratio_[:nc]
    print(ratio_frag)
    print([sum(ratio_frag[:i+1]) for i in range(nc)])
    pca_link = PCA().fit(link_fps)
    ratio_link = pca_link.explained_variance_ratio_[:nc]
    print(ratio_link)
    print([sum(ratio_link[:i+1]) for i in range(nc)])

    t4 = dt.now()
    print(t4-t3)


if __name__ == '__main__':
    main()