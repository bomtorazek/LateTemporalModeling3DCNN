import numpy as np 
import time
from two_stream_bert import utils, treg
from tqdm import tqdm
import torch

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

def train(ul_train_loader, model, criterion, optimizer, epoch, modality, args, length, input_size, writer,scheduler, val_loader = None, saveLocation = False):
    print(f"start {epoch} train")
 
    batch_time = utils.AverageMeter()
    losses_u = utils.AverageMeter()
    mask_probs = utils.AverageMeter()
    top1 = utils.AverageMeter()
    top3 = utils.AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    optimizer.zero_grad()
    Lu_mini_batch_classification = 0.0
    mask_mini_batch_classification = 0.0
    
    if not args.save_every_eval:
        p_bar = tqdm(len(ul_train_loader))
        for batch_idx, (_, (inputs_u_w, inputs_u_s), _) in enumerate(ul_train_loader):
            if modality == "rgb" or modality == "pose":
                if "3D" in args.arch or "r2plus1d" in args.arch or 'slowfast' in args.arch:
                    inputs_u_w = inputs_u_w.view(-1, length, 3, input_size, input_size).transpose(1,2)
                    inputs_u_s = inputs_u_s.view(-1, length, 3, input_size, input_size).transpose(1,2)

            elif modality == "flow":
                if "3D" in args.arch or "r2plus1d" in args.arch:
                    inputs_u_w = inputs_u_w.view(-1, length, 2, input_size, input_size).transpose(1,2)
                    inputs_u_s = inputs_u_s.view(-1, length, 2, input_size, input_size).transpose(1,2)
            else:
                inputs_u_w = inputs_u_w.view(-1, 2*length, input_size, input_size)         
                inputs_u_s = inputs_u_s.view(-1, 2*length, input_size, input_size)         

            inputs = interleave(
                torch.cat((inputs_u_w, inputs_u_s)), 2).cuda()
            logits, _, _, _ = model(inputs)
            logits = de_interleave(logits,2) # logits >> [size*batch, class]
            logits_u_w, logits_u_s = logits.chunk(2)
            del logits

            ## ----loss
            pseudo_label = torch.softmax(logits_u_w.detach()/args.T, dim=-1) # weak aug
            max_probs, targets_u = torch.max(pseudo_label, dim=-1) 
            mask = max_probs.ge(args.threshold).float() # masking
            # print('\n',torch.argmax(logits_u_s).item(), targets_u.item())

            Lu = (criterion(logits_u_s, targets_u,
                                    reduction='none') * mask).mean()
            Lu /= args.iter_size
            Lu.backward()
            maskmean = mask.mean() /args.iter_size

            Lu_mini_batch_classification += Lu.data.item()
            mask_mini_batch_classification += maskmean.data.item()

        
            ## -----optimizer step
            if (batch_idx+1) % args.iter_size == 0:
                # compute gradient and do SGD step
                optimizer.step()
                scheduler.step()
                model.zero_grad()
                losses_u.update(Lu_mini_batch_classification, args.batch_size)
                mask_probs.update(mask_mini_batch_classification, args.batch_size)
                
                batch_time.update(time.time() - end)
                end = time.time()
                Lu_mini_batch_classification = 0
                mask_mini_batch_classification = 0
            

                p_bar.set_description("Eph: {epoch}/{epochs:4}. It: {batch:4}/{iter:4}. Lu: {loss_u:.6f}. Mask: {mask:.2f}. ".format(
                    epoch=epoch + 1,
                    epochs=args.epochs,
                    batch=batch_idx + 1,
                    iter=len(ul_train_loader),
                    loss_u=losses_u.avg,
                    mask=mask_probs.avg))
                p_bar.update() 
                break
        p_bar.close()
        writer.add_scalar('data/unlabeled_loss_training', losses_u.avg, epoch)
    
    else:
        best_acc1 = 0 
        unlabeled_iter = iter(ul_train_loader)
        p_bar = tqdm(range(args.eval_step))

        for batch_idx in range(len(ul_train_loader)):
            model.train()
            try:
                _, (inputs_u_w, inputs_u_s), _ = unlabeled_iter.next()
            except:
                unlabeled_iter = iter(unlabeled_trainloader)
                _, (inputs_u_w, inputs_u_s), _ = unlabeled_iter.next()

            if modality == "rgb" or modality == "pose":
                if "3D" in args.arch or "r2plus1d" in args.arch or 'slowfast' in args.arch:
                    inputs_u_w = inputs_u_w.view(-1, length, 3, input_size, input_size).transpose(1,2)
                    inputs_u_s = inputs_u_s.view(-1, length, 3, input_size, input_size).transpose(1,2)

            elif modality == "flow":
                if "3D" in args.arch or "r2plus1d" in args.arch:
                    inputs_u_w = inputs_u_w.view(-1, length, 2, input_size, input_size).transpose(1,2)
                    inputs_u_s = inputs_u_s.view(-1, length, 2, input_size, input_size).transpose(1,2)
            else:
                inputs_u_w = inputs_u_w.view(-1, 2*length, input_size, input_size)         
                inputs_u_s = inputs_u_s.view(-1, 2*length, input_size, input_size)         

            inputs = interleave(
                torch.cat((inputs_u_w, inputs_u_s)), 2).cuda()
            logits, _, _, _ = model(inputs)
            logits = de_interleave(logits,2) # logits >> [size*batch, class]
            logits_u_w, logits_u_s = logits.chunk(2)
            del logits

            ## ----loss
            pseudo_label = torch.softmax(logits_u_w.detach()/args.T, dim=-1) # weak aug
            max_probs, targets_u = torch.max(pseudo_label, dim=-1) 
            mask = max_probs.ge(args.threshold).float() # masking
            # print('\n',torch.argmax(logits_u_s).item(), targets_u.item(), mask.mean())

            Lu = (criterion(logits_u_s, targets_u,
                                    reduction='none') * mask).mean()
            Lu /= args.iter_size
            Lu.backward()
            maskmean = mask.mean() /args.iter_size

            Lu_mini_batch_classification += Lu.data.item()
            mask_mini_batch_classification += maskmean.data.item()

        
            ## -----optimizer step
            if (batch_idx+1) % args.iter_size == 0:
                # compute gradient and do SGD step
                optimizer.step()
                scheduler.step()
                model.zero_grad()
                losses_u.update(Lu_mini_batch_classification, args.batch_size)
                mask_probs.update(mask_mini_batch_classification, args.batch_size)
                
                batch_time.update(time.time() - end)
                end = time.time()
                Lu_mini_batch_classification = 0
                mask_mini_batch_classification = 0
            

                p_bar.set_description("Eph: {epoch}/{epochs:4}. It: {batch:4}/{iter:4}. Lu: {loss_u:.6f}. Mask: {mask:.2f}. ".format(
                    epoch=epoch + 1,
                    epochs=args.epochs,
                    batch=batch_idx + 1,
                    iter=len(ul_train_loader),
                    loss_u=losses_u.avg,
                    mask=mask_probs.avg))
                p_bar.update() 
            if (batch_idx+1) % args.eval_step == 0:
                real_iter = epoch*args.eval_step + batch_idx+1
                acc1,acc3,lossClassification = validate(val_loader, model, criterion, modality, args, length, input_size)
                args.writer.add_scalar('data/top1_validation', acc1,real_iter)
                args.writer.add_scalar('data/classification_loss_validation', lossClassification, real_iter)
                args.writer.add_scalar('data/unlabeled_loss_training', losses_u.avg, real_iter)
                
                # remember best acc@1 and save checkpoint
                is_best = acc1 >= best_acc1 #FIXME best acc is not global
                best_acc1 = max(acc1, best_acc1) 

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
    
    
        print(' * * acc@1 {top1.avg:.3f} acc@3 {top3.avg:.3f} Classification Loss {lossClassification.avg:.4f}\n' 
              .format(top1=top1, top3=top3, lossClassification=lossesClassification))

    return top1.avg, top3.avg, lossesClassification.avg