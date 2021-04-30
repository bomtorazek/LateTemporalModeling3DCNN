#FIXME
def get_dataset(args):
    if args.dataset=='ucf101':
        dataset='./datasets/ucf101_frames'
    elif args.dataset=='hmdb51':
        dataset='./datasets/hmdb51_frames'
    elif args.dataset=='smtV2':
        dataset='./datasets/smtV2_frames'
    elif args.dataset=='window':
        dataset='./datasets/window_frames'
    elif args.dataset=='cvpr':
        dataset='/data/hyeokjae/results/UG2-2021/optical_flow/tv-l1/Track2.1/raw'
    elif args.dataset == 'cvpr_sid':
        dataset='/data/hyeokjae/results/UG2-2021/optical_flow/tv-l1/Track2.1/sid'
    elif args.dataset == 'cvpr_sid_sm':
        dataset='/data/hyeokjae/results/UG2-2021/optical_flow/tv-l1/Track2.1/sid_smoothing'
    elif args.dataset == 'cvpr_sid_gic':
        dataset='/data/hyeokjae/results/UG2-2021/optical_flow/tv-l1/Track2.1/sid_gic'
    elif args.dataset=='cvpr_le':
        dataset='./datasets/cvpr_le_frames'
    else:
        print("No convenient dataset entered, exiting....")
        return 0

    return dataset