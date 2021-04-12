import video_transforms
from utils.RandAugment_fixmatch import RandAugmentMC

def get_size(args):
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

    return input_size, width, height


def get_data_stat(args):
    modality = args.arch.split('_')[0]

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
        raise NameError("No such modality. Only rgb and flow supported.")
    
    return length, modality, is_color, scale_ratios, clip_mean, clip_std




class TransformFixMatch(object):
    def __init__(self, input_size, scale_ratios, randaug, totensor, normalize):
        randaug = randaug.split('_')
        rand_n = int(randaug[0])
        rand_m =int(randaug[1])

        if totensor == 1:    
            self.weak = video_transforms.Compose([
                    video_transforms.MultiScaleCrop((input_size, input_size), scale_ratios),
                    video_transforms.RandomHorizontalFlip(),
                    video_transforms.ToTensor(),
                    normalize,
                ])
            self.strong = video_transforms.Compose([
                    video_transforms.MultiScaleCrop((input_size, input_size), scale_ratios),
                    video_transforms.RandomHorizontalFlip(),
                    RandAugmentMC(rand_n, rand_m),
                    video_transforms.ToTensor(),
                    normalize,
                ])
        else:
            self.weak = video_transforms.Compose([
                    video_transforms.MultiScaleCrop((input_size, input_size), scale_ratios),
                    video_transforms.RandomHorizontalFlip(),
                    video_transforms.ToTensor2(),
                    normalize,
                ])
            self.strong = video_transforms.Compose([
                    video_transforms.MultiScaleCrop((input_size, input_size), scale_ratios),
                    video_transforms.RandomHorizontalFlip(),
                    RandAugmentMC(rand_n, rand_m),
                    video_transforms.ToTensor2(),
                    normalize,
                ])

    def __call__(self, x):
        return self.weak(x), self.strong(x)

def get_transforms(input_size, scale_ratios, clip_mean, clip_std, args):
    normalize = video_transforms.Normalize(mean=clip_mean, std=clip_std)

    if "3D" in args.arch and not ('I3D' in args.arch):
        train_transform = video_transforms.Compose([
                video_transforms.MultiScaleCrop((input_size, input_size), scale_ratios),
                video_transforms.RandomHorizontalFlip(),
                video_transforms.ToTensor2(),
                normalize,
            ])

        unlabeled_train_transform = TransformFixMatch(input_size, scale_ratios,args.randaug, 2, normalize)
    
        val_transform = video_transforms.Compose([
                video_transforms.CenterCrop((input_size)),
                video_transforms.ToTensor2(),
                normalize,
            ])
    else:
        train_transform = video_transforms.Compose([
                video_transforms.MultiScaleCrop((input_size, input_size), scale_ratios),
                video_transforms.RandomHorizontalFlip(),
                video_transforms.ToTensor(),
                normalize,
            ])

        unlabeled_train_transform = TransformFixMatch(input_size,scale_ratios, args.randaug, 1, normalize)
        
        val_transform = video_transforms.Compose([
                video_transforms.CenterCrop((input_size)),
                video_transforms.ToTensor(),
                normalize,
            ])
    

    return train_transform, unlabeled_train_transform, val_transform