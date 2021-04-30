#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from tensorboardX import SummaryWriter

import datasets
import math
from two_stream_bert import option, data_prep, build, utils, learn_contrastive, data_config, optimization, lr_scheduler

def main():
    best_acc1 = 0

    cudnn.benchmark = True
    args = option.get_args()
    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu
    input_size, width, height = data_prep.get_size(args)

    saveLocation = args.save_dir + "/" + args.dataset + "_" + args.arch + "_split" + str(args.split) +  '_contrastive_' + str(args.threshold) +'_transfer_'
    if args.model_transfer == '':
        saveLocation += 'False'
    else:
        saveLocation += 'True'
    if args.randaug:
        saveLocation += '_randaug_' + args.randaug 
    saveLocation +=  '_total_step_' + str(args.total_steps)
    saveLocation += '_lambda_' + str(args.lambda_ic) + '_' + str(int(args.lambda_gc)) + '_Tau_' + str(args.T)

    if not os.path.exists(saveLocation):
        os.makedirs(saveLocation)
    args.writer = SummaryWriter(saveLocation)
    print("Saving everything to directory %s." % (saveLocation))


    ## -------------------------- define dataset specification

    dataset = data_config.get_dataset(args)
    length, modality, is_color, scale_ratios, clip_mean, clip_std = data_prep.get_data_stat(args) #length = 64 or 32

    train_transform, _, val_transform = data_prep.get_transforms(input_size, scale_ratios, clip_mean, clip_std, args)
    train_transform_half, _, _ = data_prep.get_transforms(input_size, scale_ratios, clip_mean[:length*3//2], clip_std[:length*3//2], args)

    ## ------------------------  data loading
    train_setting_file = "l_train_%s_split%d.txt" % (modality, args.split)
    train_split_file = os.path.join(args.settings, args.dataset, train_setting_file)
    ul_train_setting_file = "u_train_%s_split%d.txt" % (modality, args.split)
    ul_train_split_file = os.path.join(args.settings, args.dataset, ul_train_setting_file)
    val_setting_file = "val_%s_split%d.txt" % (modality, args.split)
    val_split_file = os.path.join(args.settings, args.dataset, val_setting_file)
    if not os.path.exists(train_split_file):
        print("No split file exists in %s directory. Preprocess the dataset first" % (train_split_file))
    if not os.path.exists(val_split_file):
        print("No split file exists in %s directory. Preprocess the dataset first" % (val_split_file))
    if not os.path.exists(ul_train_split_file):
        print("No split file exists in %s directory. Preprocess the dataset first" % (ul_train_split_file))
    

    train_dataset = datasets.__dict__[args.dataset](root=dataset, source=train_split_file, phase="train", modality=modality, is_color=is_color,
                                                    new_length=length, new_width=width, new_height=height, video_transform=train_transform, num_segments=args.num_seg)
    ul_train_dataset = datasets.__dict__[args.dataset](root=dataset, source=ul_train_split_file, phase="ul_train", modality=modality, is_color=is_color,
                                                    new_length=length, new_width=width, new_height=height, video_transform=train_transform, num_segments=args.num_seg)
    ul_train_dataset_half1 = datasets.__dict__[args.dataset](root=dataset, source=ul_train_split_file, phase="ul_train", modality=modality, is_color=is_color,
                                                    new_length=length//2, new_width=width, new_height=height, video_transform=train_transform_half, num_segments=args.num_seg)
    ul_train_dataset_half2 = datasets.__dict__[args.dataset](root=dataset, source=ul_train_split_file, phase="ul_train", modality=modality, is_color=is_color,
                                                    new_length=length//2, new_width=width, new_height=height, video_transform=train_transform_half, num_segments=args.num_seg)
    val_dataset = datasets.__dict__[args.dataset](root=dataset, source=val_split_file, phase="val", modality=modality, is_color=is_color,
                                                  new_length=length, new_width=width, new_height=height, video_transform=val_transform, num_segments=args.num_seg)
    
    print('{} samples found, {} train samples, {} unlabeled train samples, and {} test samples.'.format(len(val_dataset)+len(train_dataset) +len(ul_train_dataset), len(train_dataset),len(ul_train_dataset), len(val_dataset)))

    ul_batch = int(args.batch_size * 0.5)  
    train_loader = torch.utils.data.DataLoader(train_dataset, sampler=RandomSampler(train_dataset), batch_size=args.batch_size, num_workers=args.workers, pin_memory=True, drop_last=True)
    ul_train_loader = torch.utils.data.DataLoader(ul_train_dataset, sampler= SequentialSampler(ul_train_dataset), batch_size= ul_batch,num_workers=args.workers, pin_memory=True, drop_last= False)
    ul_train_loader_half1 = torch.utils.data.DataLoader(ul_train_dataset_half1, sampler= SequentialSampler(ul_train_dataset_half1), batch_size= ul_batch*2,num_workers=args.workers, pin_memory=True, drop_last= True)
    ul_train_loader_half2 = torch.utils.data.DataLoader(ul_train_dataset_half2, sampler= SequentialSampler(ul_train_dataset_half2), batch_size= ul_batch*2,num_workers=args.workers, pin_memory=True, drop_last= True)
    val_loader = torch.utils.data.DataLoader(val_dataset, sampler=SequentialSampler(val_dataset), batch_size=args.batch_size, num_workers=args.workers, pin_memory=True)


    ## ------------------------- create model and optimizer
    if args.evaluate:
        print("Building validation model ... ")
        model = build.build_model_validate(args)
        optimizer = optimization.get_optimizer(model, args)
    elif args.contine: 
        model, startEpoch, optimizer, best_acc1 = build.build_model_continue(args)
        for param_group in optimizer.param_groups:
            lr = param_group['lr']
        print("Continuing with best accuracy: %.3f and start epoch %d and lr: %f" %(best_acc1, startEpoch, lr))
    else: #training ##
        print("Building model ... ")
        if args.model_transfer == '':
            model = build.build_model(args)
        else:
            model = build.build_model(args, finetune=True)
        ## ----optimizer
        optimizer = optimization.get_optimizer(model, args)
        args.eval_step = int(1513/(args.batch_size * args.mu))
        args.eval_step = args.eval_step - args.eval_step %8
        args.epochs = math.ceil(args.total_steps / args.eval_step) # 2**20 / 1024
        startEpoch = 0
    print("Model %s is loaded. " % (args.arch))
    
    ##---------------------- define loss function (criterion) and learning rate scheduler
    criterion = F.cross_entropy
    criterion_bce = nn.BCEWithLogitsLoss()
    
    # scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, verbose=True)
    scheduler = lr_scheduler.get_cosine_schedule_with_warmup(
        optimizer, args.warmup, args.total_steps)

 
    model.zero_grad()
    for epoch in range(startEpoch, args.epochs):
        if args.save_every_eval:
            best_acc1= learn_contrastive.train(train_loader, ul_train_loader, ul_train_loader_half1, ul_train_loader_half2, model, criterion, criterion_bce, optimizer, epoch, modality, args, length, input_size, args.writer, scheduler, val_loader, saveLocation,best_acc1)
        else:
            learn_contrastive.train(train_loader, ul_train_loader, ul_train_loader_half1, ul_train_loader_half2, model, criterion, criterion_bce, optimizer, epoch, modality, args, length, input_size, args.writer, scheduler)

          
        # evaluate on validation set
        acc1 = 0.0
        lossClassification = 0
        if not args.save_every_eval:
            if (epoch + 1) % args.save_freq == 0:
                acc1,acc3,lossClassification = learn_contrast.validate(val_loader, model, criterion, criterion_bce, modality, args, length, input_size)
                
                args.writer.add_scalar('data/top1_validation', acc1, epoch)
                args.writer.add_scalar('data/classification_loss_validation', lossClassification, epoch)
        
            # remember best acc@1 and save checkpoint
            is_best = acc1 >= best_acc1
            best_acc1 = max(acc1, best_acc1) 

            if (epoch + 1) % args.save_freq == 0:
                checkpoint_name = "%03d_%s" % (epoch , "checkpoint.pth.tar")
                if is_best:
                    print("Model son iyi olarak kaydedildi")
                    utils.save_checkpoint({
                        'epoch': epoch + 1,
                        'arch': args.arch,
                        'state_dict': model.state_dict(),
                        'best_acc1': best_acc1,
                        'optimizer' : optimizer.state_dict(),
                    }, is_best, checkpoint_name, saveLocation)
    
    checkpoint_name = "%03d_%s" % (epoch + 1, "checkpoint.pth.tar")
    utils.save_checkpoint({
        'epoch': epoch + 1,
        'arch': args.arch,
        'state_dict': model.state_dict(),
        'best_acc1': best_acc1,
        'optimizer' : optimizer.state_dict(),
    }, is_best, checkpoint_name, saveLocation)
    writer.export_scalars_to_json("./all_scalars.json")
    writer.close()

if __name__ == '__main__':
    main()
