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

def train(train_loader,ul_train_loader, model, criterion, optimizer, epoch, modality, args, length, input_size, writer,scheduler,  val_loader = None, saveLocation = None, best_acc1= None):
    print(f"start {epoch} train")
 
    batch_time = utils.AverageMeter()
    losses_x = utils.AverageMeter()
    losses_u = utils.AverageMeter()
    mask_probs = utils.AverageMeter()
    top1 = utils.AverageMeter()
    top3 = utils.AverageMeter()

    if args.nu > 0:
        labeled_iter = iter(train_loader)
    if args.mu > 0:
        unlabeled_iter = iter(ul_train_loader)
    
    # switch to train mode
    model.train()

    end = time.time()
    optimizer.zero_grad()
    Lu_mini_batch_classification = 0.0
    Lx_mini_batch_classification = 0.0
    mask_mini_batch_classification = 0.0
    acc_mini_batch = 0.0
    acc_mini_batch_top3 = 0.0
    totalSamplePerIter=0

    p_bar = tqdm(range(args.eval_step))
    for batch_idx in range(args.eval_step):
        if args.save_every_eval:
            model.train()
        if args.nu > 0:
            for l_idx in range(args.nu):
                try:
                    _, inputs_x, targets_x = labeled_iter.next()
        
                except:
                    labeled_iter = iter(train_loader)
                    _, inputs_x, targets_x = labeled_iter.next()

                if modality == "rgb" or modality == "pose":
                    if "3D" in args.arch or "r2plus1d" in args.arch or 'slowfast' in args.arch:
                        inputs_x = inputs_x.view(-1, length, 3, input_size, input_size).transpose(1,2)
                elif modality == "flow":
                    if "3D" in args.arch or "r2plus1d" in args.arch:
                        inputs_x = inputs_x.view(-1, length, 2, input_size, input_size).transpose(1,2)
                else:
                    inputs_x = inputs_x.view(-1, 2*length, input_size, input_size)

                ## --temporal
                lam = 1.0
                rand_index = None
                r = np.random.rand(1)
                if r < args.treg_mix_prob and args.mix_type != "None":
                    inputs_x, targets_x, lam, rand_index = treg.mix_regularization(inputs_x, targets_x, args, input_size, length)
                
                targets_x = targets_x.cuda()
                inputs_x = inputs_x.cuda()
                logits_x, _, _, _ = model(inputs_x)

                # -----------------Temporal Regularization
                if r < args.treg_mix_prob and args.mix_type in ["cutmix", "framecutmix", "cubecutmix", "mixup", "fademixup", "mcutmix"]:
                    Lx = criterion(logits_x, targets_x, reduction='mean')* lam + \
                        criterion(logits_x, targets_x[rand_index]) * (1. - lam)
                else:
                    Lx = criterion(logits_x, targets_x, reduction='mean')

                Lx /= args.iter_size  #*args.nu
                Lx.backward()
                ##  ------ train acc for HMDB51
                acc1, acc3 = utils.accuracy(logits_x.data, targets_x, topk=(1, 3))
                acc_mini_batch += acc1.item()
                acc_mini_batch_top3 += acc3.item()
                totalSamplePerIter +=  logits_x.size(0)
        if args.mu > 0:
            for ul_idx in range(args.mu * args.batch_size):
                try:
                    _, (inputs_u_w, inputs_u_s), _ = unlabeled_iter.next()
                except:
                    unlabeled_iter = iter(ul_train_loader)
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

                Lu = (criterion(logits_u_s, targets_u,
                                        reduction='none') * mask).mean()
                Lu /= args.iter_size  #* args.mu
                Lu.backward()
                maskmean = mask.mean() / (args.iter_size * args.mu)

                Lu_mini_batch_classification += Lu.data.item()
                if args.nu != 0:
                    Lx_mini_batch_classification += Lx.data.item()
                mask_mini_batch_classification += maskmean.data.item()
                if args.nu ==0:
                    totalSamplePerIter +=  logits_u_w.size(0) *2
       
        ## -----optimizer step
        if (batch_idx+1) % args.iter_size == 0:
            # compute gradient and do SGD step
            optimizer.step()
            scheduler.step()
            model.zero_grad()
            losses_x.update(Lx_mini_batch_classification,totalSamplePerIter)
            losses_u.update(Lu_mini_batch_classification, totalSamplePerIter)
            mask_probs.update(mask_mini_batch_classification, totalSamplePerIter)
            
            top1.update(acc_mini_batch/args.iter_size, totalSamplePerIter)
            top3.update(acc_mini_batch_top3/args.iter_size, totalSamplePerIter)
            batch_time.update(time.time() - end)
            end = time.time()
            Lx_mini_batch_classification = 0
            Lu_mini_batch_classification = 0
            mask_mini_batch_classification = 0
            acc_mini_batch = 0
            acc_mini_batch_top3 = 0.0
            totalSamplePerIter = 0.0

        p_bar.set_description("Eph: {epoch}/{epochs:4}. It: {batch:4}/{iter:4}. LR: {lr:.4f}. Lx: {loss_x:.4f}. Lu: {loss_u:.6f}. Mask: {mask:.2f}. ".format(
            epoch=epoch + 1,
            epochs=args.epochs,
            batch=batch_idx + 1,
            iter=args.eval_step,
            lr=scheduler.get_last_lr()[0],
            loss_x=losses_x.avg,
            loss_u=losses_u.avg,
            mask=mask_probs.avg))
        p_bar.update()   

        if args.save_every_eval and (batch_idx+1) % 4 == 0:
            real_iter = epoch*args.eval_step + batch_idx+1
            acc1,acc3,lossClassification = validate(val_loader, model, criterion, modality, args, length, input_size)
            args.writer.add_scalar('data/top1_validation', acc1,real_iter)
            args.writer.add_scalar('data/classification_loss_validation', lossClassification, real_iter)
            args.writer.add_scalar('data/unlabeled_loss_training', losses_u.avg, real_iter)
            
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



    print(' * Epoch: {epoch} HMDB51 acc@1 {top1.avg:.3f} acc@3 {top3.avg:.3f} \n'
          .format(epoch = epoch, top1=top1, top3=top3))
    p_bar.close()
    writer.add_scalar('data/labeled_loss_training', losses_x.avg, epoch)
    writer.add_scalar('data/top1_training', top1.avg, epoch)

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