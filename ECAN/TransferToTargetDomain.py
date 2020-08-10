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
from torch.utils.tensorboard import SummaryWriter

from Loss import Entropy, DANN, CDAN, HAFN, SAFN, Reweighted_MMD
from Utils import *

parser = argparse.ArgumentParser(description='Domain adaptation for Expression Classification')

parser.add_argument('--Log_Name', type=str, help='Log Name')
parser.add_argument('--OutputPath', type=str, help='Output Path')
parser.add_argument('--Backbone', type=str, default='ResNet50', choices=['ResNet18', 'ResNet50', 'VGGNet', 'MobileNet'])
parser.add_argument('--Resume_Model', type=str, help='Resume_Model', default='None')
parser.add_argument('--GPU_ID', default='0', type=str, help='CUDA_VISIBLE_DEVICES')

parser.add_argument('--useMMD', type=str2bool, default=False, help='whether to use DAN Loss')
parser.add_argument('--Gamma', type=float, default=0.3, help='weight of alpha mmd loss(default: 0.3)')
parser.add_argument('--Lambda', type=float, default=0.01, help='weight of conditional mmd loss(default: 0.01)')

parser.add_argument('--faceScale', type=int, default=112, help='Scale of face (default: 112)')
parser.add_argument('--sourceDataset', type=str, default='RAF', choices=['RAF', 'AFED', 'WFED', 'FER2013'])
parser.add_argument('--targetDataset', type=str, default='CK+', choices=['RAF', 'CK+', 'JAFFE', 'MMI', 'Oulu-CASIA', 'SFEW', 'FER2013', 'ExpW', 'AFED', 'WFED'])
parser.add_argument('--train_batch_size', type=int, default=64, help='input batch size for training (default: 64)')
parser.add_argument('--test_batch_size', type=int, default=64, help='input batch size for testing (default: 64)')
parser.add_argument('--useMultiDatasets', type=str2bool, default=False, help='whether to use MultiDataset')

parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--epochs', type=int, default=10, help='number of epochs to train (default: 10)')
parser.add_argument('--momentum', type=float, default=0.5,  help='SGD momentum (default: 0.5)')
parser.add_argument('--weight_decay', type=float, default=0.0005, help='SGD weight decay (default: 0.0005)')

parser.add_argument('--isTest', type=str2bool, default=False, help='whether to test model')
parser.add_argument('--class_num', type=int, default=7, help='number of class (default: 7)')

def Train(args, model, train_source_dataloader, train_target_dataloader, optimizer, epoch, writer):
    """Train."""

    model.train()
    torch.autograd.set_detect_anomaly(True)

    acc, prec, recall = [AverageMeter() for i in range(7)], [AverageMeter() for i in range(7)], [AverageMeter() for i in range(7)]
    loss, global_cls_loss, local_cls_loss, mmd_loss = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
    data_time, batch_time = AverageMeter(), AverageMeter()

    # Decay Learn Rate per Epoch
    if epoch <= 20:
        args.lr = 0.001
    elif epoch <= 40:
        args.lr = 0.0001
    else:
        args.lr = 0.00001

    optimizer, lr = lr_scheduler_withoutDecay(optimizer, lr=args.lr)

    source_dataset, target_dataset = train_source_dataloader.dataset, train_target_dataloader.dataset
    global source_index, target_index, source_info, target_info
    if epoch%20==1:
        source_index, target_index, source_info, target_info = GetIndexFromDataset(model, source_dataset, target_dataset)
    num_iter = max(source_info[0], target_info[0]) // args.train_batch_size

    end = time.time()  
    for batch_index in range(num_iter):
        
        data_source, landmark_source, label_source = GetDataFromDataset(source_dataset, source_index, args.train_batch_size)
        data_target, landmark_target, label_target = GetDataFromDataset(target_dataset, target_index, args.train_batch_size)
        data_time.update(time.time()-end)

        data_source, landmark_source, label_source = data_source.cuda(), landmark_source.cuda(), label_source.cuda()
        data_target, landmark_target, label_target = data_target.cuda(), landmark_target.cuda(), label_target.cuda()

        # Forward Propagation
        end = time.time()
        feature, output, loc_output = model(torch.cat((data_source, data_target), 0), torch.cat((landmark_source, landmark_target), 0))
        batch_time.update(time.time()-end)

        # Compute Loss
        global_cls_loss_ = nn.CrossEntropyLoss()(output.narrow(0, 0, data_source.size(0)), label_source)
        local_cls_loss_ = nn.CrossEntropyLoss()(loc_output.narrow(0, 0, data_source.size(0)), label_source)

        alpha_mmd_loss_, conditional_mmd_loss_ = Reweighted_MMD(feature, source_info, target_info)

        loss_ = global_cls_loss_ + local_cls_loss_

        if args.useMMD:
            loss_+=args.Gamma * alpha_mmd_loss_ + args.Lambda * conditional_mmd_loss_

        # Back Propagation
        optimizer.zero_grad()
        
        with torch.autograd.detect_anomaly():
            loss_.backward()

        optimizer.step()

        # Compute accuracy, precision and recall
        Compute_Accuracy(args, output.narrow(0, 0, data_source.size(0)), label_source, acc, prec, recall)

        # Log loss
        loss.update(float(loss_.cpu().data.item()))
        global_cls_loss.update(float(global_cls_loss_.cpu().data.item()))
        local_cls_loss.update(float(local_cls_loss_.cpu().data.item()))
        mmd_loss.update(float((args.Gamma * alpha_mmd_loss_ + args.Lambda * conditional_mmd_loss_).cpu().data.item()) if args.useMMD else 0)

        writer.add_scalar('Glocal_Cls_Loss', float(global_cls_loss_.cpu().data.item()), num_iter*(epoch-1)+batch_index)
        writer.add_scalar('Local_Cls_Loss', float(local_cls_loss_.cpu().data.item()), num_iter*(epoch-1)+batch_index)
        writer.add_scalar('MMD_Loss', float((args.Gamma * alpha_mmd_loss_ + args.Lambda * conditional_mmd_loss_).cpu().data.item()) if args.useMMD else 0, num_iter*(epoch-1)+batch_index)

        end = time.time()

    AccuracyInfo, acc_avg, prec_avg, recall_avg, f1_avg = Show_Accuracy(acc, prec, recall, args.class_num)

    writer.add_scalar('Accuracy', acc_avg, epoch)
    writer.add_scalar('Precision', prec_avg, epoch)
    writer.add_scalar('Recall', recall_avg, epoch)
    writer.add_scalar('F1', f1_avg, epoch)

    LoggerInfo = '''
    [Tain]: 
    Epoch {0}
    Data Time {data_time.sum:.4f} ({data_time.avg:.4f})
    Batch Time {batch_time.sum:.4f} ({batch_time.avg:.4f})
    Learning Rate {1}\n'''.format(epoch, lr, data_time=data_time, batch_time=batch_time)

    LoggerInfo+=AccuracyInfo

    LoggerInfo+='''    Acc_avg {0:.4f} Prec_avg {1:.4f} Recall_avg {2:.4f} F1_avg {3:.4f}
    Total Loss {loss:.4f} Global Cls Loss {global_cls_loss:.4f} Local Cls Loss {local_cls_loss:.4f} MMD Loss {mmd_loss:.4f}'''.format(acc_avg, prec_avg, recall_avg, f1_avg,
                                                                                                                                      loss=loss.avg, 
                                                                                                                                      global_cls_loss=global_cls_loss.avg, 
                                                                                                                                      local_cls_loss=local_cls_loss.avg, 
                                                                                                                                      mmd_loss=mmd_loss.avg if args.useMMD else 0)
                                                                                                
    print(LoggerInfo)

def Test(args, model, test_source_dataloader, test_target_dataloader, Best_Accuracy, Best_Recall, epoch, writer):
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
            feature, output, loc_output = model(input, landmark)
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

    # Test on Target Domain
    acc, prec, recall = [AverageMeter() for i in range(7)], [AverageMeter() for i in range(7)], [AverageMeter() for i in range(7)]
    loss, data_time, batch_time =  AverageMeter(), AverageMeter(), AverageMeter()

    end = time.time()
    for batch_index, (input, landmark, target) in enumerate(iter_target_dataloader):
        data_time.update(time.time()-end)

        input, landmark, target = input.cuda(), landmark.cuda(), target.cuda()
        
        with torch.no_grad():
            end = time.time()
            feature, output, loc_output = model(input, landmark)
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

    # Save Checkpoints
    if recall_avg > Best_Recall:
        Best_Recall = recall_avg
        print('[Save] Best Recall: %.4f.' % Best_Recall)

        if isinstance(model, nn.DataParallel):
            torch.save(model.module.state_dict(), os.path.join(args.OutputPath, '{}_Recall.pkl'.format(args.Log_Name)))
        else:
            torch.save(model.state_dict(), os.path.join(args.OutputPath, '{}_Recall.pkl'.format(args.Log_Name)))
    
    if acc_avg > Best_Accuracy:
        Best_Accuracy = acc_avg
        print('[Save] Best Accuracy: %.4f.' % Best_Accuracy)

        if isinstance(model, nn.DataParallel):
            torch.save(model.module.state_dict(), os.path.join(args.OutputPath, '{}_Accuracy.pkl'.format(args.Log_Name)))
        else:
            torch.save(model.state_dict(), os.path.join(args.OutputPath, '{}_Accuracy.pkl'.format(args.Log_Name)))

    return Best_Accuracy, Best_Recall

def main():
    """Main."""

    # Parse Argument
    args = parser.parse_args()
    torch.manual_seed(args.seed)

    # Experiment Information
    print('Log Name: %s' % args.Log_Name)
    print('Output Path: %s' % args.OutputPath)
    print('Backbone: %s' % args.Backbone)
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

        if args.useMMD:
            print('Gamma of MMD Loss: %f' % args.Gamma)
            print('Lambda of MMD Loss: %f' % args.Lambda)

    print('================================================')

    print('Number of classes : %d' % args.class_num)

    print('================================================')

    # Bulid Dataloder
    print("Buliding Train and Test Dataloader...")
    train_source_dataloader = BulidDataloader(args, flag1='train', flag2='source')
    train_target_dataloader = BulidDataloader(args, flag1='train', flag2='target')
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
    Best_Accuracy, Best_Recall = 0, 0

    # Running Experiment
    print("Run Experiment...")
    writer = SummaryWriter(os.path.join(args.OutputPath, args.Log_Name))

    source_index, target_index, source_info, target_info = None, None, None, None
    for epoch in range(1, args.epochs + 1):

        if not args.isTest:
            Train(args, model, train_source_dataloader, train_target_dataloader, optimizer, epoch, writer)  

        Best_Accuracy, Best_Recall = Test(args, model, test_source_dataloader, test_target_dataloader, Best_Accuracy, Best_Recall, epoch, writer)

    writer.close()

if __name__ == '__main__':
    main()
