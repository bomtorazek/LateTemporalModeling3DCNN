import numpy as np 
import time
from two_stream_bert import utils, treg
from tqdm import tqdm
import torch


class Trainer():
    def __init__(self,train_loader,val_loader, model, criterion, optimizer, modality, args, length, input_size, writer, scaler):
        self.train_loader = train_loader
        self.val_loader = val_loader 
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.modality = modality
        self.args = args
        self.length = length
        self.input_size = input_size
        self.writer = writer
        self.scaler = scaler

    def train(self,epoch):
        print(f"start {epoch} train")
        batch_time = utils.AverageMeter()
        lossesClassification = utils.AverageMeter()
        top1 = utils.AverageMeter()
        top3 = utils.AverageMeter()
        
        # switch to train mode
        self.model.train()

        end = time.time()
        self.optimizer.zero_grad()
        loss_mini_batch_classification = 0.0
        acc_mini_batch = 0.0
        acc_mini_batch_top3 = 0.0
        totalSamplePerIter=0

        for i, (_, inputs, targets) in enumerate(tqdm(self.train_loader)):
            if self.modality == "rgb" or self.modality == "pose":
                if "3D" in self.args.arch or "r2plus1d" in self.args.arch or 'slowfast' in self.args.arch:
                    inputs = inputs.view(-1, self.length, 3, self.input_size, self.input_size).transpose(1,2)
            elif self.modality == "flow":
                if "3D" in self.args.arch or "r2plus1d" in self.args.arch:
                    inputs = inputs.view(-1, self.length, 2, self.input_size, self.input_size).transpose(1,2)
                else:
                    inputs = inputs.view(-1, 2*self.length, self.input_size, self.input_size)          
            elif self.modality == "both":
                inputs = inputs.view(-1, 5*self.length, self.input_size, self.input_size)
                
            # ---------- Temporal Regularization
            lam = 1.0
            rand_index = None
            r = np.random.rand(1)
            if r < self.args.treg_mix_prob and self.args.mix_type != "None":
                inputs, targets, lam, rand_index = treg.mix_regularization(inputs, targets, self.args, self.input_size, self.length)
            # ----------

            # ---------- precision ---------
            if self.args.half_precision:
                inputs = inputs.cuda().half()
            else:
                inputs = inputs.cuda()
            targets = targets.cuda()

            if self.args.amp:
                with torch.cuda.amp.autocast():
                    output, input_vectors, sequenceOut, maskSample = self.model(inputs)
                    # Compute the loss.
                    # ------------------------Temporal Regularization
                    if r < self.args.treg_mix_prob and self.args.mix_type in ["cutmix", "framecutmix", "cubecutmix", "mixup", "fademixup", "mcutmix"]:
                        lossClassification = self.criterion(output, targets) * lam + self.criterion(output, targets[rand_index]) * (1. - lam)
                    else:
                        lossClassification = self.criterion(output, targets)
                    
                    lossClassification = lossClassification / self.args.iter_size
                    #totalLoss=lossMSE
                    totalLoss=lossClassification 
                    #totalLoss = lossMSE + lossClassification 
                    loss_mini_batch_classification += lossClassification.data.item()
            else:
                output, input_vectors, sequenceOut, maskSample = self.model(inputs)

                # Compute the loss.
                # ------------------------Temporal Regularization
                if r < self.args.treg_mix_prob and self.args.mix_type in ["cutmix", "framecutmix", "cubecutmix", "mixup", "fademixup", "mcutmix"]:
                    lossClassification = self.criterion(output, targets) * lam + self.criterion(output, targets[rand_index]) * (1. - lam)
                else:
                    lossClassification = self.criterion(output, targets)
                
                lossClassification = lossClassification / self.args.iter_size
                #totalLoss=lossMSE
                totalLoss=lossClassification 
                #totalLoss = lossMSE + lossClassification 
                loss_mini_batch_classification += lossClassification.data.item()


            if self.args.amp:
                self.scaler.scale(totalLoss).backward()
            else:
                totalLoss.backward()

            totalSamplePerIter +=  output.size(0)
            acc1, acc3 = utils.accuracy(output.data, targets, topk=(1, 3))
            acc_mini_batch += acc1.item()
            acc_mini_batch_top3 += acc3.item()

            if (i+1) % self.args.iter_size == 0:
                # compute gradient and do SGD step
                if self.args.amp:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                self.optimizer.zero_grad()

                lossesClassification.update(loss_mini_batch_classification, totalSamplePerIter)
                top1.update(acc_mini_batch/self.args.iter_size, totalSamplePerIter)
                top3.update(acc_mini_batch_top3/self.args.iter_size, totalSamplePerIter)
                batch_time.update(time.time() - end)
                end = time.time()
                loss_mini_batch_classification = 0
                acc_mini_batch = 0
                acc_mini_batch_top3 = 0.0
                totalSamplePerIter = 0.0
                
            if (i+1) % self.args.print_freq == 0:
                print('[%d] time: %.3f loss: %.4f' %(i,batch_time.avg,lossesClassification.avg))
            
        print(' * Epoch: {epoch} acc@1 {top1.avg:.3f} acc@3 {top3.avg:.3f} Classification Loss {lossClassification.avg:.4f}\n'
            .format(epoch = epoch, top1=top1, top3=top3, lossClassification=lossesClassification))
            
        self.writer.add_scalar('data/classification_loss_training', lossesClassification.avg, epoch)
        self.writer.add_scalar('data/top1_training', top1.avg, epoch)
        self.writer.add_scalar('data/top3_training', top3.avg, epoch)


    def validate(self):
        batch_time = utils.AverageMeter()
        lossesClassification = utils.AverageMeter()
        top1 = utils.AverageMeter()
        top3 = utils.AverageMeter()
        # switch to evaluate mode
        self.model.eval()

        end = time.time()
        with torch.no_grad():
            for i, (_, inputs, targets) in enumerate(self.val_loader):
                if self.modality == "rgb" or self.modality == "pose":
                    if "3D" in self.args.arch or "r2plus1d" in self.args.arch or 'slowfast' in self.args.arch:
                        inputs = inputs.view(-1, self.length, 3, self.input_size, self.input_size).transpose(1,2)
                elif self.modality == "flow":
                    if "3D" in self.args.arch or "r2plus1d" in self.args.arch:
                        inputs = inputs.view(-1, self.length, 2, self.input_size, self.input_size).transpose(1,2)
                    else:
                        inputs = inputs.view(-1, 2*self.length, self.input_size, self.input_size)
                elif self.modality == "both":
                    inputs = inputs.view(-1, 5*self.length, self.input_size, self.input_size)
                    
                if self.args.half_precision:
                    inputs = inputs.cuda().half()
                else:
                    inputs = inputs.cuda()
                targets = targets.cuda()
        
                # compute output

                if self.args.amp:
                    with torch.cuda.amp.autocast():
                        output, input_vectors, sequenceOut, _ = self.model(inputs)
                        lossClassification = self.criterion(output, targets)
                else:
                    output, input_vectors, sequenceOut, _ = self.model(inputs)
                    lossClassification = self.criterion(output, targets)
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