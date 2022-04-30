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

import utils
from utils import *
from ChemGraph import *
from LogMetric import Logger, AverageMeter

# Original authors are: (some parts are modified)
__author__ = "Pau Riba, Anjan Dutta"
__email__ = "priba@cvc.uab.cat, adutta@cvc.uab.cat"

def train(train_loader, model, cuda, criterion, optimizer, epoch, evaluation, log_interval, logger=None):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    error_ratio = AverageMeter()
    
    # switch to train mode                                                                                                                                                                                                                                 
    model.train()
    end = time.time()
    for i, (g, h, e, target) in enumerate(train_loader):
        # Prepare input data                                                                                                                                                                            
        if cuda:
            g, h, e, target = g.cuda(), h.cuda(), e.cuda(), target.cuda()
        g, h, e, target = Variable(g), Variable(h), Variable(e), Variable(target)
        # Measure data loading time                                                                                                                                                                      
        data_time.update(time.time() - end)
        optimizer.zero_grad()
        # Compute output                                                                                                                                                                                       
        output = model(g, h, e)
        train_loss = criterion(output, target)
        # Logs                                                                                                                                                                                                                                             
        losses.update(train_loss.data, g.size(0))
        error_ratio.update(evaluation(output, target).data, g.size(0))
        # compute gradient and do SGD step                                                                                                                                                                                                                 
        train_loss.backward()
        optimizer.step()
        # Measure elapsed time                                                                                                                                                                                                                             
        batch_time.update(time.time() - end)
        end = time.time()
        if i % log_interval == 0 and i > 0:
            print('Epoch #{0}, batch {1} of {2}\n\t'
                  'Time {batch_time.val:.3f} (average => {batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} (average => {data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} (average => {loss.avg:.4f})\t'
                  'Error Ratio {err.val:.4f} (average => {err.avg:.4f})'
                  .format(epoch, i, len(train_loader), batch_time=batch_time,
                          data_time=data_time, loss=losses, err=error_ratio),flush=True)

    logger.log_value('train_epoch_loss', losses.avg)
    logger.log_value('train_epoch_error_ratio', error_ratio.avg)
    print('Epoch: #{0} Avg Error Ratio {err.avg:.3f}; Average Loss {loss.avg:.3f}; Avg Time x Batch {b_time.avg:.3f}'
          .format(epoch, err=error_ratio, loss=losses, b_time=batch_time),flush=True)

def validate(val_loader, model, criterion, evaluation, cuda, log_interval, logger=None, epoch=""):
    batch_time = AverageMeter()
    losses = AverageMeter()
    error_ratio = AverageMeter()

    # switch to evaluate mode                                                                                                                                                                                                                              
    model.eval()
    end = time.time()
    for i, (g, h, e, target) in enumerate(val_loader):
        # Prepare input data                                                                                                                                                                                                                               
        if cuda:
            g, h, e, target = g.cuda(), h.cuda(), e.cuda(), target.cuda()
        g, h, e, target = Variable(g), Variable(h), Variable(e), Variable(target)
        # Compute output                                                                                                                                                                                                                                  
        output = model(g, h, e)
        predictions = output.tolist()
        ground_truth = target.tolist()
        if not os.path.exists("./predictions"):
            os.mkdir("./predictions")
        with open(f"./predictions/analysis{epoch}.csv", "a") as f:
            for i, val in enumerate(predictions):
                f.write("{}, {}\n".format(round(val[0],4), round(ground_truth[i][0],4)))
        # Logs                                                                                                                                                                                                                                            
        losses.update(criterion(output, target).data, g.size(0))
        error_ratio.update(evaluation(output, target).data, g.size(0))
        # measure elapsed time                                                                                                                                                                                                                             
        batch_time.update(time.time() - end)
        end = time.time()
        if i % log_interval == 0 and i > 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Error Ratio {err.val:.4f} ({err.avg:.4f})'
                  .format(i, len(val_loader), batch_time=batch_time,
                          loss=losses, err=error_ratio),flush=True)            
    print(' * Average Error Ratio {err.avg:.3f}; Average Loss {loss.avg:.3f}'
          .format(err=error_ratio, loss=losses),flush=True)
    if logger is not None:
        logger.log_value('test_epoch_loss', losses.avg)
        logger.log_value('test_epoch_error_ratio', error_ratio.avg)

    return error_ratio.avg, output, target

def look_for_best(path, model, optimizer):
    checkpoint_dir = path
    best_model_file = os.path.join(checkpoint_dir, 'model_best.pth')
    if not os.path.isdir(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    if os.path.isfile(best_model_file):
        print("=> loading best model '{}'".format(best_model_file),flush=True)
        checkpoint = torch.load(best_model_file)
        start_epoch = checkpoint['epoch']
        best_acc1 = checkpoint['best_er1']    
        model.load_state_dict(checkpoint['state_dict'])    
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded best model '{}' (epoch {})".format(best_model_file, checkpoint['epoch']),flush=True)
    else:
        print("=> no best model found at '{}'".format(best_model_file),flush=True)
