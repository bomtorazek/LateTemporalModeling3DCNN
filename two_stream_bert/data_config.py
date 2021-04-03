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
        if args.light_enhanced:
            dataset='./datasets/cvpr_frames_enhanced'
        else:
            dataset='./datasets/cvpr_frames'
    elif args.dataset=='cvpr_le':
        dataset='./datasets/cvpr_le_frames'
    else:
        print("No convenient dataset entered, exiting....")
        return 0

    return dataset