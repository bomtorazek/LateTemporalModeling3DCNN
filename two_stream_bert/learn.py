import numpy as np 
import time
from two_stream_bert import utils, treg
from tqdm import tqdm
import torch

def train(train_loader, model, criterion, optimizer, epoch, modality, args, length, input_size, writer):
    print(f"start {epoch} train")
    batch_time = utils.AverageMeter()
    lossesClassification = utils.AverageMeter()
    top1 = utils.AverageMeter()
    top3 = utils.AverageMeter()
    
    # switch to train mode
    model.train()

    end = time.time()
    optimizer.zero_grad()
    loss_mini_batch_classification = 0.0
    acc_mini_batch = 0.0
    acc_mini_batch_top3 = 0.0
    totalSamplePerIter=0

    for i, (_, inputs, targets) in enumerate(tqdm(train_loader)):
        if modality == "rgb" or modality == "pose":
            if "3D" in args.arch or "r2plus1d" in args.arch or 'slowfast' in args.arch:
                inputs = inputs.view(-1, length, 3, input_size, input_size).transpose(1,2)
        elif modality == "flow":
            if "3D" in args.arch or "r2plus1d" in args.arch:
                inputs = inputs.view(-1, length, 2, input_size, input_size).transpose(1,2)
            else:
                inputs = inputs.view(-1, 2*length, input_size, input_size)          
        elif modality == "both":
            inputs = inputs.view(-1, 5*length, input_size, input_size)
            
        # ---------- Inductive Priors for Label Change
        # target --> [Run, Sit, Stand, Turn, Walk, Wave]
        # Rewind Sit --> Stand / Rewind Stand --> Sit / Rewind Turn --> Turn / Rewind Wave --> Wave

        if args.reverse_aug:
            r = np.random.rand(1)
            if r < 0.5:
                for k in range(len(inputs)): # inputs = [2, 3, 64, 112, 112] / targets = [2]
                    #print(inputs[k].size(), targets[k]) 
                    if targets[k] == 1:
                        targets[k] = 2
                        #inputs[k] = inputs[k][:, ::-1, :, :]
                        inputs[k] = torch.flip(inputs[k], (1,))
                    elif targets[k] == 2:
                        targets[k] = 1
                        inputs[k] = torch.flip(inputs[k], (1,))
                    elif targets[k] == 3 or targets[k] == 5:
                        inputs[k] = torch.flip(inputs[k], (1,))
                    else:
                        pass
        # ----------
        
        # ---------- Temporal Regularization
        lam = 1.0
        rand_index = None
        r = np.random.rand(1)
        if r < args.treg_mix_prob and args.mix_type != "None":
            inputs, targets, lam, rand_index = treg.mix_regularization(inputs, targets, args, input_size, length)
        # ----------


        if args.half_precision:
            inputs = inputs.cuda().half()
        else:
            inputs = inputs.cuda()
        targets = targets.cuda()

        output, input_vectors, sequenceOut, maskSample = model(inputs)
        
        acc1, acc3 = utils.accuracy(output.data, targets, topk=(1, 3))
        acc_mini_batch += acc1.item()
        acc_mini_batch_top3 += acc3.item()
        
        # Compute the loss.
        # ------------------------Temporal Regularization
        if r < args.treg_mix_prob and args.mix_type in ["cutmix", "framecutmix", "cubecutmix", "mixup", "fademixup", "mcutmix"]:
            lossClassification = criterion(output, targets) * lam + criterion(output, targets[rand_index]) * (1. - lam)
        else:
            lossClassification = criterion(output, targets)
        
        lossClassification = lossClassification / args.iter_size
        
        #totalLoss=lossMSE
        totalLoss=lossClassification 
        #totalLoss = lossMSE + lossClassification 
        loss_mini_batch_classification += lossClassification.data.item()
        totalLoss.backward()
        totalSamplePerIter +=  output.size(0)
        
        if (i+1) % args.iter_size == 0:
            # compute gradient and do SGD step
            optimizer.step()
            optimizer.zero_grad()
            lossesClassification.update(loss_mini_batch_classification, totalSamplePerIter)
            top1.update(acc_mini_batch/args.iter_size, totalSamplePerIter)
            top3.update(acc_mini_batch_top3/args.iter_size, totalSamplePerIter)
            batch_time.update(time.time() - end)
            end = time.time()
            loss_mini_batch_classification = 0
            acc_mini_batch = 0
            acc_mini_batch_top3 = 0.0
            totalSamplePerIter = 0.0
            
        if (i+1) % args.print_freq == 0:
            print('[%d] time: %.3f loss: %.4f' %(i,batch_time.avg,lossesClassification.avg))
          
    print(' * Epoch: {epoch} acc@1 {top1.avg:.3f} acc@3 {top3.avg:.3f} Classification Loss {lossClassification.avg:.4f}\n'
          .format(epoch = epoch, top1=top1, top3=top3, lossClassification=lossesClassification))
          
    writer.add_scalar('data/classification_loss_training', lossesClassification.avg, epoch)
    writer.add_scalar('data/top1_training', top1.avg, epoch)
    writer.add_scalar('data/top3_training', top3.avg, epoch)


def validate(val_loader, model, criterion, modality, args, length, input_size):
    batch_time = utils.AverageMeter()
    lossesClassification = utils.AverageMeter()
    top1 = utils.AverageMeter()
    top3 = utils.AverageMeter()
    # switch to evaluate mode
    model.eval()

    end = time.time()
    with torch.no_grad():
        for i, (_, inputs, targets) in enumerate(val_loader):
            if modality == "rgb" or modality == "pose":
                if "3D" in args.arch or "r2plus1d" in args.arch or 'slowfast' in args.arch:
                    inputs = inputs.view(-1, length, 3, input_size, input_size).transpose(1,2)
            elif modality == "flow":
                if "3D" in args.arch or "r2plus1d" in args.arch:
                    inputs = inputs.view(-1, length, 2, input_size, input_size).transpose(1,2)
                else:
                    inputs = inputs.view(-1, 2*length, input_size, input_size)
            elif modality == "both":
                inputs = inputs.view(-1, 5*length, input_size, input_size)
                
            if args.half_precision:
                inputs = inputs.cuda().half()
            else:
                inputs = inputs.cuda()
            targets = targets.cuda()
    
            # compute output
            output, input_vectors, sequenceOut, _ = model(inputs)
            
            lossClassification = criterion(output, targets)
    
            # measure accuracy and record loss
            acc1, acc3 = utils.accuracy(output.data, targets, topk=(1, 3))
            
            lossesClassification.update(lossClassification.data.item(), output.size(0))
            
            top1.update(acc1.item(), output.size(0))
            top3.update(acc3.item(), output.size(0))
    
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
    
    
        print(' * * acc@1 {top1.avg:.3f} acc@3 {top3.avg:.3f} Classification Loss {lossClassification.avg:.4f}\n' 
              .format(top1=top1, top3=top3, lossClassification=lossesClassification))

    return top1.avg, top3.avg, lossesClassification.avg