import torch

from opt.AdamW import AdamW
from torch.optim import SGD
# from adamp import AdamP 
# from madgrad import MADGRAD

""" Reference 
AdamP: https://github.com/clovaai/AdamP / https://arxiv.org/abs/2006.08217
MADGRAD: https://github.com/facebookresearch/madgrad / https://arxiv.org/abs/2101.11075

how to use? just pip3 install adamp, madgrad

"""

def get_optimizer(model, args):
    if args.optimizer == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr= args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == "AdamW":
        optimizer = AdamW(model.parameters(), lr= args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == "AdamP":
        optimizer = AdamP(model.parameters(), lr= args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == "MADGRAD":
        optimizer = MADGRAD(model.parameters(), lr= args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == "SGD":
        no_decay = ['bias', 'bn']
        grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(
                nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
            {'params': [p for n, p in model.named_parameters() if any(
                nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = SGD(grouped_parameters, lr=args.lr,
                            momentum=0.9, nesterov=args.nesterov)
    
    else:
        raise NameError("No such optimizer!! Only Adam, AdamW, AdamP, MADGRAD supported")
    return optimizer