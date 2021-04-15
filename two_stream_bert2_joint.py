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
from two_stream_bert import option, data_prep, build, utils, learn_concat, learn_alter, data_config, optimization, lr_scheduler

def main():
    best_acc1 = 0

    cudnn.benchmark = True
    args = option.get_args()
    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu
    input_size, width, height = data_prep.get_size(args)

    saveLocation = args.save_dir + "/" + args.dataset + "_" + args.arch + "_split" + str(args.split) + "_mixtype_" + str(args.mix_type) + '_joint_' + str(args.threshold) + str(args.use_ema)
    if args.randaug:
        saveLocation += '_randaug_'+args.randaug


    if not os.path.exists(saveLocation):
        os.makedirs(saveLocation)
    args.writer = SummaryWriter(saveLocation)
    print("Saving everything to directory %s." % (saveLocation))


    ## -------------------------- define dataset specification

    dataset = data_config.get_dataset(args)
    length, modality, is_color, scale_ratios, clip_mean, clip_std = data_prep.get_data_stat(args)

    train_transform, ul_train_transform, val_transform = data_prep.get_transforms(input_size, scale_ratios, clip_mean, clip_std, args)

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
                                                    new_length=length, new_width=width, new_height=height, video_transform=ul_train_transform, num_segments=args.num_seg)
    val_dataset = datasets.__dict__[args.dataset](root=dataset, source=val_split_file, phase="val", modality=modality, is_color=is_color,
                                                  new_length=length, new_width=width, new_height=height, video_transform=val_transform, num_segments=args.num_seg)
    
    print('{} samples found, {} train samples, {} unlabeled train samples, and {} test samples.'.format(len(val_dataset)+len(train_dataset) |len(ul_train_dataset), len(train_dataset),len(ul_train_dataset), len(val_dataset)))

    ul_batch = int(args.batch_size * 0.5) if '64f' in args.arch else args.batch_size
    train_sampler = RandomSampler 
    train_loader = torch.utils.data.DataLoader(train_dataset, sampler=train_sampler(train_dataset), batch_size=args.batch_size, num_workers=args.workers, pin_memory=True, drop_last=True)
    ul_train_loader = torch.utils.data.DataLoader(ul_train_dataset, sampler= train_sampler(ul_train_dataset), batch_size= ul_batch,num_workers=args.workers, pin_memory=True, drop_last= True)
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
        model = build.build_model(args)
        ## ----optimizer
        optimizer = optimization.get_optimizer(model, args)
        args.epochs = math.ceil(args.total_steps / args.eval_step) # 2**20 / 1024
        startEpoch = 0
    print("Model %s is loaded. " % (args.arch))
    
    ##---------------------- define loss function (criterion) and learning rate scheduler
    criterion = F.cross_entropy
    
    # scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, verbose=True)
    scheduler = lr_scheduler.get_cosine_schedule_with_warmup(
        optimizer, args.warmup, args.total_steps)


    if args.evaluate:
        if '64f' in args.arch:
            acc1,acc3, lossClassification = learn_alter.validate(val_loader, model, criterion, modality, args, length, input_size)
        else:
            acc1,acc3, lossClassification = learn_concat.validate(val_loader, model, criterion, modality, args, length, input_size)
        return

 
    model.zero_grad()
    for epoch in range(startEpoch, args.epochs):
        if '64f' in args.arch:
            learn_alter.train(train_loader, ul_train_loader, model, criterion, optimizer, epoch, modality, args, length, input_size, args.writer, scheduler)
        else:
            learn_concat.train(train_loader, ul_train_loader, model, criterion, optimizer, epoch, modality, args, length, input_size, args.writer, scheduler)
            
        # evaluate on validation set
        acc1 = 0.0
        lossClassification = 0
        if (epoch + 1) % args.save_freq == 0:
            if '64f' in args.arch:
                acc1,acc3,lossClassification = learn_alter.validate(val_loader, model, criterion, modality, args, length, input_size)
            else:
                acc1,acc3,lossClassification = learn_concat.validate(val_loader, model, criterion, modality, args, length, input_size)

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
