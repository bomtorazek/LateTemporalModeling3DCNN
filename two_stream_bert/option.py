import argparse
import models

model_names = sorted(name for name in models.__dict__
    if not name.startswith("__")
    and callable(models.__dict__[name]))

def get_args():
    parser = argparse.ArgumentParser(description='PyTorch Two-Stream Action Recognition')


    ### Dataset
    #parser.add_argument('--data', metavar='DIR', default='./datasets/ucf101_frames',help='path to dataset')
    parser.add_argument('--settings', metavar='DIR', default='./datasets/settings',
                        help='path to dataset setting files')
    #parser.add_argument('--modality', '-m', metavar='MODALITY', default='rgb',
    #                    choices=["rgb", "flow"], help='modality: rgb | flow')
    parser.add_argument('--dataset', '-d', default='hmdb51',
                        choices=["ucf101", "hmdb51", "smtV2", "window", "semi_cvpr", "cvpr_le"], help='dataset: ucf101 | hmdb51 | smtV2')
    parser.add_argument('-s', '--split', default=1, type=int, metavar='S', help='which split of data to work on (default: 1)')
    parser.add_argument('-j', '--workers', default=2, type=int, metavar='N', help='number of data loading workers (default: 2)')
    parser.add_argument('--arch', '-a', default='rgb_resneXt3D64f101_bert10_FRMB', choices=model_names,
                        help='model architecture: ' + ' | '.join(model_names) + ' (default: rgb_resneXt3D64f101_bert10_FRMB)')
    parser.add_argument('--mu', default=3, type=int,
                        help='coefficient of unlabeled batch size')
    parser.add_argument('--nu', default=1, type=int,
                    help='coefficient of labeled batch size')
    parser.add_argument('--lambda-u', default=1, type=float,
                        help='coefficient of unlabeled loss')
    parser.add_argument('--T', default=1, type=float,
                        help='pseudo label temperature')
    parser.add_argument('--threshold', default=0.95, type=float,
                        help='pseudo label threshold')
    parser.add_argument('--light_enhanced', action='store_true', default=False)
    parser.add_argument('--save_dir', metavar='DIR', default='./checkpoint',help='path to save checkpoints')


    ### Training
    parser.add_argument('--epochs', default=200, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--total-steps', default=2**20, type=int,
                        help='number of total steps to run')
    parser.add_argument('--eval-step', default=1024, type=int,
                        help='number of eval steps to run')
    parser.add_argument('-b', '--batch-size', default=1, type=int,
                        metavar='N', help='mini-batch size (default: 8)')
    parser.add_argument('--iter-size', default=16, type=int,
                        metavar='I', help='iter size to reduce memory usage (default: 16)')
    parser.add_argument('--optimizer', default='AdamW', choices=['Adam', 'AdamW', 'AdamP', 'MADGRAD', 'SGD'])
    parser.add_argument('--lr', '--learning-rate', default=1e-5, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--warmup', default=0, type=float,
                        help='warmup epochs (unlabeled data based)')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum (default: 0.9)')
    parser.add_argument('--nesterov', action='store_true', default=True,
                        help='use nesterov momentum')
    parser.add_argument('--use-ema', action='store_true', 
                        help='use EMA model')
    parser.add_argument('--ema-decay', default=0.999, type=float,
                        help='EMA decay rate')
    parser.add_argument('--weight-decay', '--wd', default=1e-3, type=float,
                        metavar='W', help='weight decay (default: 1e-3)')
    parser.add_argument('--print-freq', default=400, type=int,
                        metavar='N', help='print frequency (default: 400)')
    parser.add_argument('--save-freq', default=1, type=int,
                        metavar='N', help='save frequency (default: 1)')
    parser.add_argument('--save-every-eval', action = 'store_true',default = False)
    parser.add_argument('--num-seg', default=1, type=int,
                        metavar='N', help='Number of segments in dataloader (default: 1)')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true', help='evaluate model on validation set')
    parser.add_argument('-c', '--continue', dest='contine', action='store_true', help='continue training model')
    parser.add_argument('--gpu', default='0', type=str, help='gpu id')
    parser.add_argument('--half_precision', action='store_true', help='half precision training')
    parser.add_argument('--model-transfer', default = '', help='dir of a checkpoint to finetune')

    # For Temporal Augmentations
    parser.add_argument('--treg_mix_prob', default=1.0, type=float)
    parser.add_argument('--treg_mix_beta', default=1.0, type=float)
    parser.add_argument('--mix_type', default='None', choices=['None', 'cutmix', 'framecutmix', 'cubecutmix', 'mixup', 'fademixup', 'mcutmix', 'cutout', 'framecutout', 'cubecutout'])
    parser.add_argument('--randaug', default='', type=str,help='3_15_t for n and m respectively, add _t if randaug-t')

    args = parser.parse_args()
    return args