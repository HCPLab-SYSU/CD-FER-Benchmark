import os
import sys
import time
import tqdm
import shutil
import argparse
import subprocess
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim

import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from torch.utils.tensorboard import SummaryWriter

from Loss import LP_Loss, getCenters
from Utils import *

parser = argparse.ArgumentParser(description='Expression Classification Training')

parser.add_argument('--Log_Name', type=str, help='Log Name')
parser.add_argument('--OutputPath', type=str, help='Output Path')
parser.add_argument('--Backbone', type=str, default='ResNet50', choices=['ResNet18', 'ResNet50', 'VGGNet', 'MobileNet'])
parser.add_argument('--Resume_Model', type=str, help='Resume_Model', default='None')
parser.add_argument('--GPU_ID', default='0', type=str, help='CUDA_VISIBLE_DEVICES')

parser.add_argument('--faceScale', type=int, default=112, help='Scale of face (default: 112)')
parser.add_argument('--sourceDataset', type=str, default='RAF', choices=['RAF', 'AFED', 'WFED', 'FER2013'])
parser.add_argument('--targetDataset', type=str, default='CK+', choices=['RAF', 'CK+', 'JAFFE', 'MMI', 'Oulu-CASIA', 'SFEW', 'FER2013', 'ExpW', 'AFED', 'WFED'])
parser.add_argument('--train_batch_size', type=int, default=64, help='input batch size for training (default: 64)')
parser.add_argument('--test_batch_size', type=int, default=64, help='input batch size for testing (default: 64)')
parser.add_argument('--useMultiDatasets', type=str2bool, default=False, help='whether to use MultiDataset')

parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--epochs', type=int, default=60,help='number of epochs to train (default: 60)')
parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum (default: 0.9)')
parser.add_argument('--weight_decay', type=float, default=0.0001,help='SGD weight decay (default: 0.0001)')

parser.add_argument('--isTest', type=str2bool, default=False, help='whether to test model')

parser.add_argument('--class_num', type=int, default=7, help='number of class (default: 7)')
parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')

def Train(args, model, train_dataloader, optimizer, epoch, writer):
    """Train."""

    model.train()
    torch.autograd.set_detect_anomaly(True)

    acc, prec, recall = [AverageMeter() for i in range(7)], [AverageMeter() for i in range(7)], [AverageMeter() for i in range(7)]
    loss, fusion_loss, ic_loss, id_loss, data_time, batch_time =  AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()

    # Decay Learn Rate per Epoch
    if args.Backbone in ['ResNet18', 'ResNet50']:
        if epoch <= 20:
            args.lr = 1e-4
        elif epoch <= 40:
            args.lr = 1e-5
        else:
            args.lr = 1e-6

    elif args.Backbone == 'MobileNet':
        if epoch <= 20:
            args.lr = 1e-3
        elif epoch <= 40:
            args.lr = 1e-4
        elif epoch <= 60:
            args.lr = 1e-5
        else:
            args.lr = 1e-6

    elif args.Backbone == 'VGGNet':
        if epoch <= 30:
            args.lr = 1e-3
        elif epoch <= 60:
            args.lr = 1e-4
        elif epoch <= 70:
            args.lr = 1e-5
        else:
            args.lr = 1e-6

    end = time.time()
    for step, (input, landmark, label) in enumerate(train_dataloader):

        input, landmark, label = input.cuda(), landmark.cuda(), label.cuda()
        data_time.update(time.time()-end)

        # Forward propagation
        end = time.time()
        feature, output, IC_output, ID_output = model(input, landmark)
        batch_time.update(time.time()-end)

        # Compute Loss
        fusion_loss_ = nn.CrossEntropyLoss()(output, label) 
        ic_loss_ = nn.BCELoss()(IC_output, (label==(torch.cat((label[1:],label[0].unsqueeze(0)), 0))).float())
        id_loss_ = nn.CrossEntropyLoss()(ID_output, label)

        loss_ = fusion_loss_ + ic_loss_ + id_loss_

        # Back Propagation
        optimizer.zero_grad()
        loss_.backward()
        optimizer.step()

        # Decay Learn Rate
        optimizer, lr = lr_scheduler_withoutDecay(optimizer, lr=args.lr, weight_decay=args.weight_decay) 

        # Compute accuracy, recall and loss
        Compute_Accuracy(args, output, label, acc, prec, recall)

        # Log loss
        loss.update(float(loss_.cpu().data.item()))
        fusion_loss.update(float(fusion_loss_.cpu().data.item()))
        ic_loss.update(float(ic_loss_.cpu().data.item()))
        id_loss.update(float(id_loss_.cpu().data.item()))

        end = time.time()

    AccuracyInfo, acc_avg, prec_avg, recall_avg, f1_avg = Show_Accuracy(acc, prec, recall, args.class_num)

    writer.add_scalar('Accuracy', acc_avg, epoch)
    writer.add_scalar('Precision', prec_avg, epoch)
    writer.add_scalar('Recall', recall_avg, epoch)
    writer.add_scalar('F1', f1_avg, epoch)

    writer.add_scalar('Fusion_Loss', fusion_loss.avg, epoch)
    writer.add_scalar('IC_Loss', ic_loss.avg, epoch)
    writer.add_scalar('ID_Loss', id_loss.avg, epoch)

    LoggerInfo = '''
    [Tain]: 
    Epoch {0}
    Data Time {data_time.sum:.4f} ({data_time.avg:.4f})
    Batch Time {batch_time.sum:.4f} ({batch_time.avg:.4f})
    Learning Rate {1}\n'''.format(epoch, args.lr, data_time=data_time, batch_time=batch_time)

    LoggerInfo+=AccuracyInfo

    LoggerInfo+='''    Acc_avg {0:.4f} Prec_avg {1:.4f} Recall_avg {2:.4f} F1_avg {3:.4f}
    Total Loss {loss:.4f} Fusion Loss {fusion_loss:.4f} IC Loss {ic_loss:.4f} ID Loss {id_loss:.4f}'''.format(acc_avg, prec_avg, recall_avg, f1_avg,
                                                                                                              loss=loss.avg, 
                                                                                                              fusion_loss=fusion_loss.avg, 
                                                                                                              ic_loss=ic_loss.avg,
                                                                                                              id_loss=id_loss.avg)

    print(LoggerInfo)

def Test(args, model, test_source_dataloader, test_target_dataloader, Best_Recall, epoch, writer):
    """Test."""

    model.eval()
    torch.autograd.set_detect_anomaly(True)

    iter_source_dataloader = iter(test_source_dataloader)
    iter_target_dataloader = iter(test_target_dataloader)

    # Test on Source Domain
    acc, prec, recall = [AverageMeter() for i in range(7)], [AverageMeter() for i in range(7)], [AverageMeter() for i in range(7)]
    loss, data_time, batch_time =  AverageMeter(), AverageMeter(), AverageMeter()

    end = time.time()
    for batch_index, (input, landmark, target) in enumerate(iter_source_dataloader):
        data_time.update(time.time()-end)

        input, landmark, target = input.cuda(), landmark.cuda(), target.cuda()
        
        with torch.no_grad():
            end = time.time()
            feature, output, ic_output, id_output = model(input, landmark)
            batch_time.update(time.time()-end)
        
        loss_ = nn.CrossEntropyLoss()(output, target)

        # Compute accuracy, precision and recall
        Compute_Accuracy(args, output, target, acc, prec, recall)

        # Log loss
        loss.update(float(loss_.cpu().data.numpy()))

        end = time.time()

    AccuracyInfo, acc_avg, prec_avg, recall_avg, f1_avg = Show_Accuracy(acc, prec, recall, args.class_num)

    writer.add_scalar('Test_Recall_SourceDomain', recall_avg, epoch)
    writer.add_scalar('Test_Accuracy_SourceDomain', acc_avg, epoch)

    LoggerInfo = '''
    [Test (Source Domain)]: 
    Data Time {data_time.sum:.4f} ({data_time.avg:.4f})
    Batch Time {batch_time.sum:.4f} ({batch_time.avg:.4f})\n'''.format(data_time=data_time, batch_time=batch_time)

    LoggerInfo+=AccuracyInfo

    LoggerInfo+='''    Acc_avg {0:.4f} Prec_avg {1:.4f} Recall_avg {2:.4f} F1_avg {3:.4f}
    Loss {loss:.4f}'''.format(acc_avg, prec_avg, recall_avg, f1_avg, loss=loss.avg)

    print(LoggerInfo)

    # Save Checkpoints
    if recall_avg > Best_Recall:
        Best_Recall = recall_avg
        print('[Save] Best Recall: %.4f.' % Best_Recall)

         if isinstance(model, nn.DataParallel):
            torch.save(model.module.state_dict(), os.path.join(args.OutputPath, '{}.pkl'.format(args.Log_Name)))
        else:
            torch.save(model.state_dict(), os.path.join(args.OutputPath, '{}.pkl'.format(args.Log_Name)))

    # Test on Target Domain
    acc, prec, recall = [AverageMeter() for i in range(7)], [AverageMeter() for i in range(7)], [AverageMeter() for i in range(7)]
    loss, data_time, batch_time =  AverageMeter(), AverageMeter(), AverageMeter()

    end = time.time()
    for batch_index, (input, landmark, target) in enumerate(iter_target_dataloader):
        data_time.update(time.time()-end)

        input, landmark, target = input.cuda(), landmark.cuda(), target.cuda()
        
        with torch.no_grad():
            end = time.time()
            feature, output, ic_output, id_output = model(input, landmark)
            batch_time.update(time.time()-end)
        
        loss_ = nn.CrossEntropyLoss()(output, target)

        # Compute accuracy, precision and recall
        Compute_Accuracy(args, output, target, acc, prec, recall)

        # Log loss
        loss.update(float(loss_.cpu().data.numpy()))

        end = time.time()

    AccuracyInfo, acc_avg, prec_avg, recall_avg, f1_avg = Show_Accuracy(acc, prec, recall, args.class_num)

    writer.add_scalar('Test_Recall_TargetDomain', recall_avg, epoch)
    writer.add_scalar('Test_Accuracy_TargetDomain', acc_avg, epoch)

    LoggerInfo = '''
    [Test (Target Domain)]: 
    Data Time {data_time.sum:.4f} ({data_time.avg:.4f})
    Batch Time {batch_time.sum:.4f} ({batch_time.avg:.4f})\n'''.format(data_time=data_time, batch_time=batch_time)

    LoggerInfo+=AccuracyInfo

    LoggerInfo+='''    Acc_avg {0:.4f} Prec_avg {1:.4f} Recall_avg {2:.4f} F1_avg {3:.4f}
    Loss {loss:.4f}'''.format(acc_avg, prec_avg, recall_avg, f1_avg, loss=loss.avg)
    
    print(LoggerInfo)

    return Best_Recall

def main():
    """Main."""
 
    # Parse Argument
    args = parser.parse_args()
    torch.manual_seed(args.seed)

    # Experiment Information
    print('Log Name: %s' % args.Log_Name)
    print('Output Path: %s' % args.OutputPath)
    print('Resume Model: %s' % args.Resume_Model)
    print('CUDA_VISIBLE_DEVICES: %s' % args.GPU_ID)

    print('================================================')

    print('Use {} * {} Image'.format(args.faceScale, args.faceScale))
    print('SourceDataset: %s' % args.sourceDataset)
    print('TargetDataset: %s' % args.targetDataset)
    print('Train Batch Size: %d' % args.train_batch_size)
    print('Test Batch Size: %d' % args.test_batch_size)

    print('================================================')

    if args.isTest:
        print('Test Model.')
    else:
        print('Train Epoch: %d' % args.epochs)
        print('Learning Rate: %f' % args.lr)
        print('Momentum: %f' % args.momentum)
        print('Weight Decay: %f' % args.weight_decay)
        print('Number of classes : %d' % args.class_num)

    print('================================================')

    # Bulid Dataloder
    print("Buliding Train and Test Dataloader...")
    train_source_dataloader = BulidDataloader(args, flag1='train', flag2='source')
    test_source_dataloader = BulidDataloader(args, flag1='test', flag2='source')
    test_target_dataloader = BulidDataloader(args, flag1='test', flag2='target')
    print('Done!')

    print('================================================')

    # Bulid Model
    print('Buliding Model...')
    model = BulidModel(args)
    print('Done!')

    print('================================================')

    # Set Optimizer
    print('Buliding Optimizer...')
    param_optim = Set_Param_Optim(args, model)
    optimizer = Set_Optimizer(args, param_optim, args.lr, args.weight_decay, args.momentum)
    print('Done!')

    print('================================================')

    # Save Best Checkpoint
    Best_Recall = 0

    # Running Experiment
    print("Run Experiment...")
    writer = SummaryWriter(os.path.join(args.OutputPath, args.Log_Name))

    for epoch in range(1, args.epochs + 1):

        if not args.isTest:
            Train(args, model, train_source_dataloader, optimizer, epoch, writer)

        Best_Recall = Test(args, model, test_source_dataloader, test_target_dataloader, Best_Recall, epoch, writer)

        torch.cuda.empty_cache()

    writer.close()

if __name__ == '__main__':
    main()
