#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 10:42:00 2019

@author: esat
"""

import os
import time
import argparse
import shutil
import numpy as np
import csv


os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"


import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from tensorboardX import SummaryWriter

from torch.optim import lr_scheduler
import video_transforms
import models
import datasets
import swats
from opt.AdamW import AdamW
from utils.model_path import rgb_3d_model_path_selection
from two_stream_bert import data_config


model_names = sorted(name for name in models.__dict__
    if not name.startswith("__")
    and callable(models.__dict__[name]))

dataset_names = sorted(name for name in datasets.__all__)

parser = argparse.ArgumentParser(description='PyTorch Two-Stream Action Recognition')
parser.add_argument('--settings', metavar='DIR', default='./datasets/settings',
                    help='path to datset setting files')
parser.add_argument('--dataset', '-d', default='hmdb51',
                    choices=["ucf101", "hmdb51", "smtV2", "window", "cvpr", "semi_cvpr"],
                    help='dataset: ucf101 | hmdb51 | smtV2')

parser.add_argument('--arch', '-a', default='rgb_resneXt3D64f101_bert10_FRMB',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: rgb_resneXt3D64f101_bert10_FRMB)')
parser.add_argument('-s', '--split', default=1, type=int, metavar='S',
                    help='which split of data to work on (default: 1)')
parser.add_argument('-j', '--workers', default=2, type=int, metavar='N',
                    help='number of data loading workers (default: 2)')
parser.add_argument('-b', '--batch-size', default=8, type=int,
                    metavar='N', help='mini-batch size (default: 8)')
parser.add_argument('--num-seg', default=1, type=int,
                    metavar='N', help='Number of segments in dataloader (default: 1)')
parser.add_argument('--track', default='1', type=str)
parser.add_argument('--model-path', default = '', help='dir of a checkpoint to finetune')
parser.add_argument('--light_enhanced', action='store_true', default=False)
parser.add_argument('--gpu', default = '0', type=str, help = 'gpuid')
parser.add_argument('--tta', default=1, type=int)




smt_pretrained = False

HALF = False
training_continue = False

def main():
    global args, model,writer, length, width, height, input_size, scheduler
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu
    if '3D' in args.arch:
        if 'I3D' in args.arch or 'MFNET3D' in args.arch:
            if '112' in args.arch:
                scale = 0.5
            else:
                scale = 1
        else:
            if '224' in args.arch:
                scale = 1
            else:
                scale = 0.5
    elif 'r2plus1d' in args.arch:
        scale = 0.5
    else:
        scale = 1
        
    print('scale: %.1f' %(scale))
    
    input_size = int(224 * scale)
    width = int(340 * scale)
    height = int(256 * scale)
   
    # create model    
    print("Building validation model ... ")
    model = build_model_validate(args.model_path)
    
    if HALF:
        model.half()  # convert to half precision
        for layer in model.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.float()
    
    print("Model %s is loaded. " % (args.arch))

    dataset = data_config.get_dataset(args)
    
    cudnn.benchmark = True
    modality=args.arch.split('_')[0]
    if "3D" in args.arch or 'tsm' in args.arch or 'slowfast' in args.arch or 'r2plus1d' in args.arch:
        if '64f' in args.arch:
            length=64
        elif '32f' in args.arch:
            length=32
        else:
            length=16
    else:
        length=1
    # Data transforming
    if modality == "rgb" or modality == "pose":
        is_color = True
        scale_ratios = [1.0, 0.875, 0.75, 0.66]
        if 'I3D' in args.arch:
            if 'resnet' in args.arch:
                clip_mean = [0.45, 0.45, 0.45] * args.num_seg * length
                clip_std = [0.225, 0.225, 0.225] * args.num_seg * length
            else:
                clip_mean = [0.5, 0.5, 0.5] * args.num_seg * length
                clip_std = [0.5, 0.5, 0.5] * args.num_seg * length
            #clip_std = [0.25, 0.25, 0.25] * args.num_seg * length
        elif 'MFNET3D' in args.arch:
            clip_mean = [0.48627451, 0.45882353, 0.40784314] * args.num_seg * length
            clip_std = [0.234, 0.234, 0.234]  * args.num_seg * length
        elif "3D" in args.arch:
            clip_mean = [114.7748, 107.7354, 99.4750] * args.num_seg * length
            clip_std = [1, 1, 1] * args.num_seg * length
        elif "r2plus1d" in args.arch:
            clip_mean = [0.43216, 0.394666, 0.37645] * args.num_seg * length
            clip_std = [0.22803, 0.22145, 0.216989] * args.num_seg * length
        elif "rep_flow" in args.arch:
            clip_mean = [0.5, 0.5, 0.5] * args.num_seg * length
            clip_std = [0.5, 0.5, 0.5] * args.num_seg * length      
        elif "slowfast" in args.arch:
            clip_mean = [0.45, 0.45, 0.45] * args.num_seg * length
            clip_std = [0.225, 0.225, 0.225] * args.num_seg * length
        else:
            clip_mean = [0.485, 0.456, 0.406] * args.num_seg * length
            clip_std = [0.229, 0.224, 0.225] * args.num_seg * length
    elif modality == "pose":
        is_color = True
        scale_ratios = [1.0, 0.875, 0.75, 0.66]
        clip_mean = [0.485, 0.456, 0.406] * args.num_seg
        clip_std = [0.229, 0.224, 0.225] * args.num_seg
    elif modality == "flow":
        is_color = False
        scale_ratios = [1.0, 0.875, 0.75, 0.66]
        if 'I3D' in args.arch:
            clip_mean = [0.5, 0.5] * args.num_seg * length
            clip_std = [0.5, 0.5] * args.num_seg * length
        elif "3D" in args.arch:
            clip_mean = [127.5, 127.5] * args.num_seg * length
            clip_std = [1, 1] * args.num_seg * length        
        else:
            clip_mean = [0.5, 0.5] * args.num_seg * length
            clip_std = [0.226, 0.226] * args.num_seg * length
    elif modality == "both":
        is_color = True
        scale_ratios = [1.0, 0.875, 0.75, 0.66]
        clip_mean = [0.485, 0.456, 0.406, 0.5, 0.5] * args.num_seg * length
        clip_std = [0.229, 0.224, 0.225, 0.226, 0.226] * args.num_seg * length
    else:
        print("No such modality. Only rgb and flow supported.")

    
    normalize = video_transforms.Normalize(mean=clip_mean,
                                           std=clip_std)

    if "3D" in args.arch and not ('I3D' in args.arch):
    
        val_transform = video_transforms.Compose([
                video_transforms.CenterCrop((input_size)),
                video_transforms.ToTensor2(),
                normalize,
            ])
    else:

        val_transform = video_transforms.Compose([
                video_transforms.CenterCrop((input_size)),
                video_transforms.ToTensor(),
                normalize,
            ])

    # data loading
    val_setting_file = "test_%s_split00.txt" % (modality) #, args.split)
    #val_setting_file = "val_%s_split%d.txt" % (modality, args.split) #FIXME
    print("will test on", val_setting_file, "dataset")
    val_split_file = os.path.join(args.settings, args.dataset, val_setting_file)
    if not os.path.exists(val_split_file):
        print("No split file exists in %s directory. Preprocess the dataset first" % (args.settings))

    #root = 'cvpr_frames'!!
    
    val_dataset = datasets.__dict__[args.dataset](root=dataset,
                                                  source=val_split_file,
                                                  phase="val",
                                                  modality=modality,
                                                  is_color=is_color,
                                                  new_length=length,
                                                  new_width=width,
                                                  new_height=height,
                                                  video_transform=val_transform,
                                                  num_segments=args.num_seg,
                                                  tta=args.tta)

    print(' {} test samples.'.format(len(val_dataset)))
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    acc1,acc3= validate(val_loader, model,modality)
    
    
def build_model_validate(model_path):
    params = torch.load(model_path)
    print("model from",model_path)
    if args.dataset=='ucf101':
        model=models.__dict__[args.arch](modelPath='', num_classes=101,length=args.num_seg)
    elif args.dataset=='hmdb51':
        model=models.__dict__[args.arch](modelPath='', num_classes=51,length=args.num_seg)
    elif 'semi' in args.dataset: 
        model = models.__dict__[args.arch](modelPath='', num_classes=5, length=args.num_seg)
    elif 'cvpr' in args.dataset: 
        model = models.__dict__[args.arch](modelPath='', num_classes=6, length=args.num_seg)
    
    if torch.cuda.device_count() > 1:
        model=torch.nn.DataParallel(model) 

    model.load_state_dict(params['state_dict'])
    model.cuda()
    model.eval() 
    return model

def validate(val_loader, model,modality):
    batch_time = AverageMeter()
    top1 = AverageMeter()
    top3 = AverageMeter()
    # switch to evaluate mode
    model.eval()

    end = time.time()
    with torch.no_grad():
        pred_dict = {}
        prob_dict = {}
        softmax = nn.Softmax(dim=0)

        if args.tta != 1:
            for t in range(args.tta):
                for i, (names, inputs, targets) in enumerate(val_loader):
                    if modality == "rgb" or modality == "pose":
                        if "3D" in args.arch or "r2plus1d" in args.arch or 'slowfast' in args.arch:
                            inputs=inputs.view(-1,length,3,input_size,input_size).transpose(1,2)
                    elif modality == "flow":
                        if "3D" in args.arch or "r2plus1d" in args.arch:
                            inputs=inputs.view(-1,length,2,input_size,input_size).transpose(1,2)
                        else:
                            inputs=inputs.view(-1,2*length,input_size,input_size)
                    elif modality == "both":
                        inputs=inputs.view(-1,5*length,input_size,input_size)
                        
                    if HALF:
                        inputs = inputs.cuda().half()
                    else:
                        inputs = inputs.cuda()
                    if targets != []:
                        targets = targets.cuda()
                    
                    # compute output
                    output, input_vectors, sequenceOut, _ = model(inputs)
                    for i in range(len(names)): #FIXME
                        #save_name = int(names[i])
                        save_name = names[i]
                        if save_name in pred_dict:
                            pred_dict[save_name] += torch.argmax(output[i]).item() # if the name of files are integers
                            prob_dict[save_name] += softmax((output[i])).detach().cpu().numpy()
                        else:
                            pred_dict[save_name] = torch.argmax(output[i]).item() # if the name of files are integers
                            prob_dict[save_name] = softmax((output[i])).detach().cpu().numpy()
                    
                    # measure accuracy and record loss 
                    if targets != []: #FIXME
                        acc1, acc3 = accuracy(output.data, targets, topk=(1, 3))
                    
                        top1.update(acc1.item(), output.size(0))
                        top3.update(acc3.item(), output.size(0))

            # averaging pred dict, prob dict
            for key in pred_dict.keys():
                pred_dict[key] /= args.tta
                pred_dict[key] = int(pred_dict[key])
                prob_dict[key] /= args.tta
        
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
        else: 
            for i, (names, inputs, targets) in enumerate(val_loader):
                if modality == "rgb" or modality == "pose":
                    if "3D" in args.arch or "r2plus1d" in args.arch or 'slowfast' in args.arch:
                        inputs=inputs.view(-1,length,3,input_size,input_size).transpose(1,2)
                elif modality == "flow":
                    if "3D" in args.arch or "r2plus1d" in args.arch:
                        inputs=inputs.view(-1,length,2,input_size,input_size).transpose(1,2)
                    else:
                        inputs=inputs.view(-1,2*length,input_size,input_size)
                elif modality == "both":
                    inputs=inputs.view(-1,5*length,input_size,input_size)
                    
                if HALF:
                    inputs = inputs.cuda().half()
                else:
                    inputs = inputs.cuda()
                if targets != []:
                    targets = targets.cuda()
                
        
                # compute output
                output, input_vectors, sequenceOut, _ = model(inputs)
                for i in range(len(names)): #FIXME
                    #pred_dict[int(names[i])] = torch.argmax(output[i]).item() # if the name of files are integers
                    #prob_dict[int(names[i])] = softmax((output[i])).detach().cpu().numpy()
                    pred_dict[(names[i])] = torch.argmax(output[i]).item()
                    prob_dict[(names[i])] = softmax((output[i])).detach().cpu().numpy()


                # measure accuracy and record loss
                if targets != []:
                    acc1, acc3 = accuracy(output.data, targets, topk=(1, 3))
                
                    top1.update(acc1.item(), output.size(0))
                    top3.update(acc3.item(), output.size(0))
            
                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()


        spt = args.model_path.split('/')
        for idx, dir in enumerate(spt):
            if args.arch in dir:
                chk_idx = idx
                break
        model_name_from_path= spt[chk_idx]
        ckpt_name = spt[-1]

        # using dict, make .csv files
        with open(f'result/track{args.track}_{model_name_from_path}_{ckpt_name}_pred_tta{args.tta}.csv', 'w') as f:
            pencil = csv.writer(f) 
            pencil.writerow(['VideoID', 'Video', 'ClassID'])
            for idx, key in enumerate(sorted(pred_dict.keys())):
                pencil.writerow([idx, str(key)+'.mp4', pred_dict[key]+5]) # real class in Track2.1 # FIXME for semi

        with open(f'result/{model_name_from_path}_{ckpt_name}_prob.csv', 'w') as f:
            pencil = csv.writer(f) 
            pencil.writerow(['VideoID', 'Video', 'Run', 'Sit', 'Stand', 'Turn', 'Walk', 'Wave']) #FIXME for semi
            for idx, key in enumerate(sorted(prob_dict.keys())):  
                pencil.writerow([idx, str(key)+'.mp4', prob_dict[key][0],prob_dict[key][1],
                                                        prob_dict[key][2],prob_dict[key][3],
                                                        prob_dict[key][4],prob_dict[key][5]]) # real class



        if targets != []:    
            print(' * * acc@1 {top1.avg:.3f} acc@3 {top3.avg:.3f} \n' 
                .format(top1=top1, top3=top3))
    if targets != []:
        return top1.avg, top3.avg
    else:
        return None, None

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

        
def accuracy(output, target, topk=(1,)):
    """Computes the accuracy@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    correct = correct.contiguous()

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

if __name__ == '__main__':
    main()