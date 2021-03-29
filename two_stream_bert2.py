#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from tensorboardX import SummaryWriter

from torch.optim import lr_scheduler
import datasets
from opt.AdamW import AdamW
from two_stream_bert import option, data_prep, build, utils, learn


def main():
    best_acc1 = 0

    cudnn.benchmark = True
    args = option.get_args()
    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu
    input_size, width, height = data_prep.get_size(args)

    saveLocation = args.save_dir + "/" + args.dataset + "_" + args.arch + "_split" + str(args.split) + "_mixtype_" + str(args.mix_type)

    if not os.path.exists(saveLocation):
        os.makedirs(saveLocation)
    writer = SummaryWriter(saveLocation)
    print("Saving everything to directory %s." % (saveLocation))
   
    # create model
    if args.evaluate:
        print("Building validation model ... ")
        model = build.build_model_validate(args)
        optimizer = AdamW(model.parameters(), lr= args.lr, weight_decay=args.weight_decay)
    elif args.contine:
        model, startEpoch, optimizer, best_acc1 = build.build_model_continue(args)
        for param_group in optimizer.param_groups:
            lr = param_group['lr']
        print("Continuing with best accuracy: %.3f and start epoch %d and lr: %f" %(best_acc1, startEpoch, lr))
    else:
        print("Building model ... ")
        model = build.build_model(args)
        optimizer = AdamW(model.parameters(), lr= args.lr, weight_decay=args.weight_decay)
        startEpoch = 0
    print("Model %s is loaded. " % (args.arch))

    # convert model to half precision
    if args.half_precision:
        model.half()
        for layer in model.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.float()
        print("Model is converted to Half-Precision")

    # define loss function (criterion) and learning rate scheduler
    criterion = nn.CrossEntropyLoss().cuda()
    #criterion2 = nn.MSELoss().cuda()
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, verbose=True)

    # define dataset specification
    dataset = data_prep.get_dataset(args)
    length, modality, is_color, scale_ratios, clip_mean, clip_std = data_prep.get_data_stat(args)

    train_transform, val_transform = data_prep.get_transforms(input_size, scale_ratios, clip_mean, clip_std, args)

    # data loading
    train_setting_file = "train_%s_split%d.txt" % (modality, args.split)
    train_split_file = os.path.join(args.settings, args.dataset, train_setting_file)
    val_setting_file = "val_%s_split%d.txt" % (modality, args.split)
    val_split_file = os.path.join(args.settings, args.dataset, val_setting_file)
    if not os.path.exists(train_split_file) or not os.path.exists(val_split_file):
        print("No split file exists in %s directory. Preprocess the dataset first" % (args.settings))

    train_dataset = datasets.__dict__[args.dataset](root=dataset, source=train_split_file, phase="train", modality=modality, is_color=is_color,
                                                    new_length=length, new_width=width, new_height=height, video_transform=train_transform, num_segments=args.num_seg)
    
    val_dataset = datasets.__dict__[args.dataset](root=dataset, source=val_split_file, phase="val", modality=modality, is_color=is_color,
                                                  new_length=length, new_width=width, new_height=height, video_transform=val_transform, num_segments=args.num_seg)

    print('{} samples found, {} train samples and {} test samples.'.format(len(val_dataset)+len(train_dataset), len(train_dataset), len(val_dataset)))

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)

    if args.evaluate:
        acc1,acc3,lossClassification = learn.validate(val_loader, model, criterion, modality, args, length, input_size)
        return

    for epoch in range(startEpoch, args.epochs):
        learn.train(train_loader, model, criterion, optimizer, epoch, modality, args, length, input_size, writer)

        # evaluate on validation set
        acc1 = 0.0
        lossClassification = 0
        if (epoch + 1) % args.save_freq == 0:
            acc1,acc3,lossClassification = learn.validate(val_loader, model, criterion, modality, args, length, input_size)
            writer.add_scalar('data/top1_validation', acc1, epoch)
            writer.add_scalar('data/top3_validation', acc3, epoch)
            writer.add_scalar('data/classification_loss_validation', lossClassification, epoch)
            scheduler.step(lossClassification)

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
