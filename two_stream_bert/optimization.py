import torch

from opt.AdamW import AdamW
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
    else:
        raise NameError("No such optimizer!! Only Adam, AdamW, AdamP, MADGRAD supported")
    return optimizer