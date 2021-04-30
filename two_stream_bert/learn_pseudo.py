import numpy as np 
import time
from two_stream_bert import utils, treg
from tqdm import tqdm
import torch


def tensor_reshape(inputs, modality, args, length, input_size):
    # i =0 >> length/1, i=1 >> length//2
    for i in range(len(inputs)):
        if modality == "rgb" or modality == "pose":
            if "3D" in args.arch or "r2plus1d" in args.arch or 'slowfast' in args.arch:
                inputs[i] = inputs[i].view(-1, length//(i+1), 3, input_size, input_size).transpose(1,2)
        elif modality == "flow":
            if "3D" in args.arch or "r2plus1d" in args.arch:
                inputs[i] = inputs[i].view(-1, length//(i+1), 2, input_size, input_size).transpose(1,2)
        else:
            inputs[i] = inputs[i].view(-1, 2*length//(i+1), input_size, input_size)     
    return inputs




def interleave(x, size): #size = 2mu+1
    s = list(x.shape)
    # s=[batch *size, 3, 64, 112, 112], batch(=1) + batch*3*2 = 7
    # [ batch,size,3,64,112,112]
    # [size,batch,3,64,112,112]
    # [size*batch,3,64,112,112]

    return x.reshape([-1, size] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])

def de_interleave(x, size):
    s = list(x.shape) #[size*batch, class]
    return x.reshape([size, -1] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])

def train(ul_train_loader,  model, criterion, optimizer, epoch, modality, args, length, input_size, writer, scheduler, val_loader, saveLocation,best_acc1):
    print(f"start {epoch} train")

    batch_time = utils.AverageMeter()
    losses_x = utils.AverageMeter()
    mask_probs = utils.AverageMeter()
    top1 = utils.AverageMeter()
    top3 = utils.AverageMeter()


    unlabeled_iter = iter(ul_train_loader)

    
    # switch to train mode
    model.train()

    end = time.time()
    optimizer.zero_grad()
    Lx_mini_batch_classification = 0.0
    mask_mini_batch_classification = 0.0
    acc_mini_batch = 0.0 # for labeled training
    acc_mini_batch_top3 = 0.0
    totalSamplePerIter=0

    p_bar = tqdm(range(args.eval_step))
    for batch_idx in range(args.eval_step):
        if args.save_every_eval:
            model.train()
        
        try:
            _, inputs_x, _ = unlabeled_iter.next()

        except:
            unlabeled_iter = iter(ul_train_loader)
            _, inputs_x, _ = unlabeled_iter.next()
        
        inputs_x = tensor_reshape([inputs_x], modality, args, length, input_size )[0]
        inputs_x = inputs_x.cuda()
        logits_x, _, _, _ = model(inputs_x)
        pseudo_label = torch.softmax(logits_x.detach(), dim=-1) 
        max_probs, targets_u = torch.max(pseudo_label, dim=-1) 
        mask = max_probs.ge(args.threshold).float() # masking
        maskmean = mask.mean() / (args.iter_size )
        Lx = (criterion(logits_x, targets_u,
                                        reduction='none') * mask).mean()
        Lx /= args.iter_size 
        Lx.backward()
        Lx_mini_batch_classification += Lx.data.item()
        mask_mini_batch_classification += maskmean.data.item()
        totalSamplePerIter +=  logits_x.size(0)
       
        ## -----optimizer step
        if (batch_idx+1) % args.iter_size == 0:
            # compute gradient and do SGD step
            optimizer.step()
            scheduler.step()
            model.zero_grad()
            losses_x.update(Lx_mini_batch_classification,totalSamplePerIter)
            mask_probs.update(mask_mini_batch_classification, totalSamplePerIter)
            batch_time.update(time.time() - end)
            end = time.time()
            Lx_mini_batch_classification = 0.0
            mask_mini_batch_classification = 0.0
            totalSamplePerIter=0


        p_bar.set_description("Eph: {epoch}/{epochs:4}. It: {batch:4}/{iter:4}. LR: {lr:.4f}. Lx: {loss_x:.4f}.mask: {mask:.3f} ".format(
            epoch=epoch + 1,
            epochs=args.epochs,
            batch=batch_idx + 1,
            iter=args.eval_step,
            lr=scheduler.get_last_lr()[0],
            loss_x=losses_x.avg,
            mask=mask_probs.avg
            ))
    
        p_bar.update()   

        if args.save_every_eval and (batch_idx+1) % 16 == 0:
            real_iter = epoch*args.eval_step + batch_idx+1
            acc1,acc3,lossClassification = validate(val_loader, model, criterion, modality, args, length, input_size)
            args.writer.add_scalar('data/top1_validation', acc1,real_iter)
            args.writer.add_scalar('data/unlabeled_loss_training', losses_x.avg, real_iter)
            
            # remember best acc@1 and save checkpoint
            is_best = acc1 >= best_acc1
            best_acc1 = max(acc1, best_acc1) 
            print(acc1)
            checkpoint_name = "%03d_%s" % (real_iter , "checkpoint.pth.tar")
            if is_best:
                print("Model son iyi olarak kaydedildi")
                utils.save_checkpoint({
                    'epoch': real_iter,
                    'arch': args.arch,
                    'state_dict': model.state_dict(),
                    'best_acc1': best_acc1,
                    'optimizer' : optimizer.state_dict(),
                }, is_best, checkpoint_name, saveLocation)

    p_bar.close()

    if args.save_every_eval:
        return best_acc1


def validate(val_loader, model, criterion, modality, args, length, input_size):

    if args.use_ema:
        from models.ema import ModelEMA
        ema_model = ModelEMA(args, model, args.ema_decay)
        model = ema_model.ema

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
            top3.update(acc1.item(), output.size(0))
    
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
    
    
        # print(' * * acc@1 {top1.avg:.3f} acc@3 {top3.avg:.3f} Classification Loss {lossClassification.avg:.4f}\n' 
        #       .format(top1=top1, top3=top3, lossClassification=lossesClassification))

    return top1.avg, top3.avg, lossesClassification.avg