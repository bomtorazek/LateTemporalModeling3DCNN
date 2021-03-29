from utils.model_path import rgb_3d_model_path_selection
import models
import torch

def build_model(args):
    modality=args.arch.split('_')[0]
    if modality == "rgb":
        model_path = rgb_3d_model_path_selection(args.arch)
        #model_path = os.path.join(modelLocation,'model_best.pth.tar') 
        
    elif modality == "flow":
        model_path=''
        if "3D" in args.arch:
            if 'I3D' in args.arch:
                 model_path='./weights/flow_imagenet.pth'   
            elif '3D' in args.arch:
                 model_path='./weights/Flow_Kinetics_64f.pth'         
        #model_path = os.path.join(modelLocation,'model_best.pth.tar') 
    elif modality == "both":
        model_path='' 
        
    if args.dataset=='ucf101':
        print('model path is: %s' %(model_path))
        model = models.__dict__[args.arch](modelPath=model_path, num_classes=101,length=args.num_seg)
    elif args.dataset=='hmdb51':
        print('model path is: %s' %(model_path))
        model = models.__dict__[args.arch](modelPath=model_path, num_classes=51, length=args.num_seg)
    elif args.dataset=='smtV2':
        print('model path is: %s' %(model_path))
        model = models.__dict__[args.arch](modelPath=model_path, num_classes=174, length=args.num_seg)
    elif args.dataset=='window':
        print('model path is: %s' %(model_path))
        model = models.__dict__[args.arch](modelPath=model_path, num_classes=3, length=args.num_seg)
    elif 'cvpr' in args.dataset: # TODO for semi
        print('model path is: %s' %(model_path))
        model = models.__dict__[args.arch](modelPath=model_path, num_classes=6, length=args.num_seg)
    

    if torch.cuda.device_count() > 1:
        model=torch.nn.DataParallel(model)
    model = model.cuda()
    
    return model

def build_model_validate(args):
    modelLocation="./checkpoint/"+args.dataset+"_"+args.arch+"_split"+str(args.split)
    model_path = os.path.join(modelLocation,'model_best.pth.tar') 
    params = torch.load(model_path)
    print(modelLocation)
    if args.dataset=='ucf101':
        model=models.__dict__[args.arch](modelPath='', num_classes=101,length=args.num_seg)
    elif args.dataset=='hmdb51':
        model=models.__dict__[args.arch](modelPath='', num_classes=51,length=args.num_seg)
    elif args.dataset=='smtV2':
        print('model path is: %s' %(model_path))
        model = models.__dict__[args.arch](modelPath=model_path, num_classes=174, length=args.num_seg)
    elif args.dataset=='window':
        print('model path is: %s' %(model_path))
        model = models.__dict__[args.arch](modelPath=model_path, num_classes=3, length=args.num_seg)
    elif 'cvpr' in args.dataset: # TODO for semi
        print('model path is: %s' %(model_path))
        model = models.__dict__[args.arch](modelPath=model_path, num_classes=6, length=args.num_seg)

    if torch.cuda.device_count() > 1:
        model=torch.nn.DataParallel(model) 

    model.load_state_dict(params['state_dict'])
    model.cuda()
    model.eval() 
    return model

def build_model_continue(args):
    modelLocation="./checkpoint/"+args.dataset+"_"+args.arch+"_split"+str(args.split)
    model_path = os.path.join(modelLocation,'model_best.pth.tar') 
    params = torch.load(model_path)
    print(modelLocation)
    
    if args.dataset=='ucf101':
        model=models.__dict__[args.arch](modelPath='', num_classes=101,length=args.num_seg)
    elif args.dataset=='hmdb51':
        model=models.__dict__[args.arch](modelPath='', num_classes=51,length=args.num_seg)
    elif args.dataset=='smtV2':
        print('model path is: %s' %(model_path))
        model = models.__dict__[args.arch](modelPath=model_path, num_classes=174, length=args.num_seg)
    elif args.dataset=='window':
        print('model path is: %s' %(model_path))
        model = models.__dict__[args.arch](modelPath=model_path, num_classes=3, length=args.num_seg)
    elif 'cvpr' in args.dataset: # TODO for semi
        print('model path is: %s' %(model_path))
        model = models.__dict__[args.arch](modelPath=model_path, num_classes=6, length=args.num_seg)

    if torch.cuda.device_count() > 1:
        model=torch.nn.DataParallel(model) 
        
    model.load_state_dict(params['state_dict'])
    model = model.cuda()
    optimizer = AdamW(model.parameters(), lr= args.lr, weight_decay=args.weight_decay)
    optimizer.load_state_dict(params['optimizer'])
    
    startEpoch = params['epoch']
    best_acc = params['best_acc1']

    return model, startEpoch, optimizer, best_acc
