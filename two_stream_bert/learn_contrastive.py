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


def Normalize(x,power): # (1,5)
    norm = x.pow(power).sum(1, keepdim=True).pow(1. / power)
    out = x.div(norm + 1e-7)
    return out

def calculate_ic(pos1,pos2,neg1, args):
    batchSize = pos1.shape[0]
    ic_pos = torch.bmm(pos1.view(batchSize, 1, -1), pos2.view(batchSize, -1, 1)) # (1,1,5) * (1, 5, 1)  >> (1, 1, 1 )
    ic_pos = ic_pos.view(batchSize, 1) # (1, 1)
    ic_neg = torch.bmm(pos1.view(batchSize, 1, -1), neg1.view(batchSize, -1, 1))
    ic_neg = ic_neg.view(batchSize, 1) # (1, 1)
    out = torch.cat((ic_pos, ic_neg), dim=1) / args.T
    return out


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

def train(train_loader,ul_train_loader, ul_train_loader_half1, ul_train_loader_half2, model, criterion, optimizer, epoch, modality, args, length, input_size, writer,scheduler,  val_loader = None, saveLocation = None, best_acc1= None):
    print(f"start {epoch} train")
 
    batch_time = utils.AverageMeter()
    losses_x = utils.AverageMeter()
    losses_ic = utils.AverageMeter()
    losses_gc = utils.AverageMeter()
    mask_probs = utils.AverageMeter()
    top1 = utils.AverageMeter()
    top3 = utils.AverageMeter()

    if args.nu > 0:
        labeled_iter = iter(train_loader)
    if args.mu > 0:
        unlabeled_iter = iter(ul_train_loader)
        unlabeled_iter_half1 = iter(ul_train_loader_half1)
        unlabeled_iter_half2 = iter(ul_train_loader_half2)

    
    # switch to train mode
    model.train()

    end = time.time()
    optimizer.zero_grad()
    Lx_mini_batch_classification = 0.0
    ic_mini_batch_classification = 0.0
    gc_mini_batch_classification = 0.0
    mask_mini_batch_classification = 0.0
    acc_mini_batch = 0.0 # for labeled training
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
                
                inputs_x = tensor_reshape([inputs_x], modality, args, length, input_size )[0]
 
                ## --temporal augmentation
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

                Lx /= args.iter_size * args.nu
                Lx.backward()
                ##  ------ train acc for HMDB51
                acc1, acc3 = utils.accuracy(logits_x.data, targets_x, topk=(1, 3))
                acc_mini_batch += acc1.item()
                acc_mini_batch_top3 += acc3.item()
                totalSamplePerIter +=  logits_x.size(0)

        if args.mu > 0:
            for ul_idx in range(args.mu ):

                ##-------------- for data a!!
                try:
                    name_64a, inputs_64a, _ = unlabeled_iter.next()
                    name_32ab, inputs_32ab, _ = unlabeled_iter_half1.next()

                except:
                    unlabeled_iter = iter(ul_train_loader)
                    unlabeled_iter_half1 = iter(ul_train_loader_half1)
                    name_64a, inputs_64a, _ = unlabeled_iter.next()
                    name_32ab, inputs_32ab, _ = unlabeled_iter_half1.next()

                inputs_64a, inputs_32ab = tensor_reshape( [inputs_64a, inputs_32ab], modality, args, length, input_size)
             
                assert name_64a[0] == name_32ab[0] # check name
                inputs_64a = inputs_64a.cuda()
                inputs_32ab = inputs_32ab.cuda()
                logits_64a, _, _, _ = model(inputs_64a) # input ([1, 3, 64, 112, 112])
                logits_32ab, _,_,_ = model(inputs_32ab) # input ([2, 3, 32, 112, 112])
                logits_32a, logits_32b = logits_32ab.chunk(2)

                # logit normalization
                logits_64a = Normalize(logits_64a, 2)
                logits_32a = Normalize(logits_32a,2)
                logits_32b = Normalize(logits_32b,2) # 1,5
                
                ## instance
                # temp scaling, softmax
                out_64_a = calculate_ic(logits_64a, logits_32a, logits_32b,args) # pos, pos, neg
                L_ic_64_a =criterion(out_64_a, torch.zeros(out_64_a.size(0),dtype=torch.long, 
                                                                device= logits_64a.device), reduction='mean')
                out_32_a = calculate_ic(logits_32a, logits_64a, logits_32b,args)
                L_ic_32_a =criterion(out_32_a, torch.zeros(out_32_a.size(0),dtype=torch.long, 
                                                                device= logits_64a.device), reduction='mean')
                Lu_ic_a = args.lambda_ic*(L_ic_64_a + L_ic_32_a)/4.0 # (a,b, 32,64)

                if args.lambda_gc >0:
                    # slow 끼리 label 비교
                    _, pseudo_32_a = torch.max(logits_32a, dim=-1) 
                    _, pseudo_32_b = torch.max(logits_32b, dim=-1) 
                    print(pseudo_32_a, pseudo_32_b)
                    # 같으면 cross entropy (fast, slow)
                    # 다르면 instance negative랑 같음.


                Lu_ic_a /= args.iter_size * args.mu
                Lu_ic_a.backward()


                ##-------------- for data b!!
                try:
                    name_64b, inputs_64b, _ = unlabeled_iter.next()
                    name_32ab, inputs_32ab, _ = unlabeled_iter_half2.next()

                except:
                    unlabeled_iter = iter(ul_train_loader)
                    unlabeled_iter_half2 = iter(ul_train_loader_half2)
                    name_64b, inputs_64b, _ = unlabeled_iter.next()
                    name_32ab, inputs_32ab, _ = unlabeled_iter_half2.next()

                inputs_64b, inputs_32ab = tensor_reshape( [inputs_64b, inputs_32ab], modality, args, length, input_size)
             
                assert name_64b[0] == name_32ab[1] # check name
                inputs_64b = inputs_64b.cuda()
                inputs_32ab = inputs_32ab.cuda()
                logits_64b, _, _, _ = model(inputs_64b) # input ([1, 3, 64, 112, 112])
                logits_32ab, _,_,_ = model(inputs_32ab) # input ([2, 3, 32, 112, 112])
                logits_32a, logits_32b = logits_32ab.chunk(2)

                # logit normalization
                logits_64b = Normalize(logits_64b, 2)
                logits_32a = Normalize(logits_32a,2)
                logits_32b = Normalize(logits_32b,2) # 1,5
                
                ## instance
                # temp scaling, softmax
                out_64_b = calculate_ic(logits_64b, logits_32b, logits_32a,args) # pos, pos, neg
                L_ic_64_b =criterion(out_64_b, torch.zeros(out_64_b.size(0),dtype=torch.long, 
                                                                device= logits_64b.device), reduction='mean')
                out_32_b = calculate_ic(logits_32b, logits_64b, logits_32a,args)
                L_ic_32_b =criterion(out_32_b, torch.zeros(out_32_b.size(0),dtype=torch.long, 
                                                                device= logits_64b.device), reduction='mean')
                Lu_ic_b = args.lambda_ic*(L_ic_64_b + L_ic_32_b)/4.0
                Lu_ic_b /= args.iter_size * args.mu
                Lu_ic_b.backward()

                ic_mini_batch_classification += Lu_ic_a.item() + Lu_ic_b.item()
                if args.lambda_gc >0:
                    gc_mini_batch_classification += 0 # FIXME
                # mask_mini_batch_classification += maskmean.data.item()
                if args.nu == 0:
                    totalSamplePerIter +=  logits_32ab.size(0)
                else:
                    Lx_mini_batch_classification += Lx.data.item()
                

               
        
       
        ## -----optimizer step
        if (batch_idx+1) % args.iter_size == 0:
            # compute gradient and do SGD step
            optimizer.step()
            scheduler.step()
            model.zero_grad()
            if args.nu >0:
                losses_x.update(Lx_mini_batch_classification,totalSamplePerIter)
            losses_ic.update(ic_mini_batch_classification, totalSamplePerIter)
            if args.lambda_gc >0:
                losses_gc.update(gc_mini_batch_classification, totalSamplePerIter)
            # mask_probs.update(mask_mini_batch_classification, totalSamplePerIter)
            
            top1.update(acc_mini_batch/args.iter_size, totalSamplePerIter)
            top3.update(acc_mini_batch_top3/args.iter_size, totalSamplePerIter)
            batch_time.update(time.time() - end)
            end = time.time()
            Lx_mini_batch_classification = 0.0
            ic_mini_batch_classification = 0.0
            gc_mini_batch_classification = 0.0
            mask_mini_batch_classification = 0.0
            acc_mini_batch = 0.0 # for labeled training
            acc_mini_batch_top3 = 0.0
            totalSamplePerIter=0


        p_bar.set_description("Eph: {epoch}/{epochs:4}. It: {batch:4}/{iter:4}. LR: {lr:.4f}. Lx: {loss_x:.4f}. ic: {loss_ic:.6f}. gc: {loss_gc:.6f}. ".format(
            epoch=epoch + 1,
            epochs=args.epochs,
            batch=batch_idx + 1,
            iter=args.eval_step,
            lr=scheduler.get_last_lr()[0],
            loss_x=losses_x.avg,
            loss_ic=losses_ic.avg,
            loss_gc=losses_gc.avg))
    
        p_bar.update()   

        if args.save_every_eval and (batch_idx+1) % 8 == 0:
            real_iter = epoch*args.eval_step + batch_idx+1
            acc1,acc3,lossClassification = validate(val_loader, model, criterion, modality, args, length, input_size)
            args.writer.add_scalar('data/top1_validation', acc1,real_iter)
            args.writer.add_scalar('data/classification_loss_validation', lossClassification, real_iter)
            args.writer.add_scalar('data/unlabeled_loss_training', losses_ic.avg, real_iter)
            
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