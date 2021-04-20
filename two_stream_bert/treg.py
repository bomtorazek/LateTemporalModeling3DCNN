import torch
import numpy as np

def rand_bbox(size, lam, mix):
    T = size[2]
    W = size[3]
    H = size[4]

    if mix in ['cutmix', 'mcutmix']:
        cut_rat = np.sqrt(1. - lam)
        cut_w = np.int(W * cut_rat)
        cut_h = np.int(H * cut_rat)

        cx = np.random.randint(W)
        cy = np.random.randint(H)

        bbt1 = 0
        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbt2 = T
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)
    elif mix in ['framecutmix']:
        cut_rat = 1. - lam
        cut_t = np.int(T * cut_rat)

        ct = np.random.randint(T)

        bbt1 = np.clip(ct - cut_t // 2, 0, T)
        bbx1 = 0
        bby1 = 0
        bbt2 = np.clip(ct + cut_t // 2, 0, T)
        bbx2 = W
        bby2 = H
    else:  # spatio-temporal, cubemix
        cut_rat = np.power(1. - lam, 1. / 3.)
        cut_t = np.int(T * cut_rat)
        cut_w = np.int(W * cut_rat)
        cut_h = np.int(H * cut_rat)

        ct = np.random.randint(T)
        cx = np.random.randint(W)
        cy = np.random.randint(H)

        bbt1 = np.clip(ct - cut_t // 2, 0, T)
        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbt2 = np.clip(ct + cut_t // 2, 0, T)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbt1, bbx1, bby1, bbt2, bbx2, bby2

def rand_out(size, spatial, temporal, mix):
    T = size[2]
    W = size[3]
    H = size[4]

    if mix in ['cutout']:
        cx = np.random.randint(W - spatial)
        cy = np.random.randint(H - spatial)

        bbt1 = 0
        bbx1 = np.clip(cx, 0, W)
        bby1 = np.clip(cy, 0, H)
        bbt2 = T
        bbx2 = np.clip(cx + spatial, 0, W)
        bby2 = np.clip(cy + spatial, 0, H)
    elif mix in ['framecutout']:
        ct = np.random.randint(T - (temporal // 2))

        bbt1 = np.clip(ct, 0, T)
        bbx1 = 0
        bby1 = 0
        bbt2 = np.clip(ct + temporal // 2, 0, T)
        bbx2 = W
        bby2 = H
    else:  # cubeout
        ct = np.random.randint(T - temporal)
        cx = np.random.randint(W - spatial)
        cy = np.random.randint(H - spatial)

        bbt1 = np.clip(ct, 0, T)
        bbx1 = np.clip(cx, 0, W)
        bby1 = np.clip(cy, 0, H)
        bbt2 = np.clip(ct + temporal, 0, T)
        bbx2 = np.clip(cx + spatial, 0, W)
        bby2 = np.clip(cy + spatial, 0, H)

    return bbt1, bbx1, bby1, bbt2, bbx2, bby2

def mix_regularization(inputs, labels, cfg, input_size, length):
    lam = 0.0
    rand_index = None

    if cfg.mix_type in ["cutmix", "framecutmix", "cubecutmix", "mixup", "fademixup", "mcutmix"]:
        # Sample Mix Ratio (Lambda)
        lam = np.random.beta(cfg.treg_mix_beta, cfg.treg_mix_beta)

        # Random Mix within Batch
        rand_index = torch.randperm(inputs.size()[0]).cuda()

        if cfg.mix_type in ['cutmix', 'framecutmix', 'cubecutmix']:
            # Sample Mixing Coordinates
            bbt1, bbx1, bby1, bbt2, bbx2, bby2 = rand_bbox(inputs.size(), lam, cfg.mix_type)

            # adjust lambda to exactly match pixel ratio
            lam = 1 - ((bbt2 - bbt1) * (bbx2 - bbx1) * (bby2 - bby1) / (
                    inputs.size()[-1] * inputs.size()[-2] * inputs.size()[-3]))

            # Mix
            inputs[:, :, bbt1:bbt2, bbx1:bbx2, bby1:bby2] = inputs[rand_index, :, bbt1:bbt2, bbx1:bbx2, bby1:bby2]
        elif cfg.mix_type in ['mcutmix']:  # Moving CutMix
            # Sample Two BBoxs for Both Temporal Ends
            bbt1b, bbx1b, bby1b, bbt2b, bbx2b, bby2b = rand_bbox(inputs.size(), lam, cfg.mix_type)
            bbt1e, bbx1e, bby1e, bbt2e, bbx2e, bby2e = rand_bbox(inputs.size(), lam, cfg.mix_type)

            # adjust lambda
            h1 = (bby2b - bby1b)
            w1 = (bbx2b - bbx1b)
            tlen = inputs.size(2)
            bby_delta = ((bby2e - bby1e) - h1) / (tlen - 1)
            bbx_delta = ((bbx2e - bbx1e) - w1) / (tlen - 1)

            boxsum = tlen * h1 * w1 + (tlen * (tlen - 1) * (bbx_delta * h1 + bby_delta * w1) / 2) + (
                        (tlen - 1) * tlen * (2 * tlen - 1) * bby_delta * bbx_delta / 6)
            lam = 1 - (boxsum / (inputs.size()[-1] * inputs.size()[-2] * inputs.size()[-3]))

            # Mix
            t_weight = np.arange(tlen) / (tlen - 1)
            bbx1_t = np.around((1 - t_weight) * bbx1b + t_weight * bbx1e).astype(int)
            bbx2_t = np.around((1 - t_weight) * bbx2b + t_weight * bbx2e).astype(int)
            bby1_t = np.around((1 - t_weight) * bby1b + t_weight * bby1e).astype(int)
            bby2_t = np.around((1 - t_weight) * bby2b + t_weight * bby2e).astype(int)

            for taxis in range(inputs.size(2)):
                inputs[:, :, taxis, bbx1_t[taxis]:bbx2_t[taxis], bby1_t[taxis]:bby2_t[taxis]] = inputs[rand_index, :, taxis, bbx1_t[taxis]:bbx2_t[taxis], bby1_t[taxis]:bby2_t[taxis]]

        else:  # mixup: blending two videos
            if cfg.mix_type in ['mixup']:
                inputs = inputs * lam + inputs[rand_index] * (1. - lam)
            elif cfg.mix_type in ['fademixup']:  # temporally varying mix-up
                adj = np.random.choice([-1, 0, 1]) * np.random.uniform(0, min(lam, 1.0 - lam))
                fade = np.linspace(lam - adj, lam + adj, num=inputs.size(2))
                for taxis in range(inputs.size(2)):
                    inputs[:, :, taxis, :, :] = inputs[:, :, taxis, :, :] * fade[taxis] + inputs[rand_index, :, taxis, :, :] * (1. - fade[taxis])

        # Change Labels for Validation Accuracy
        labels = labels if lam >= 0.5 else labels[rand_index]

    elif cfg.mix_type in ['cutout', 'framecutout', 'cubecutout']:
        # Sample Out Coordinates
        bbt1, bbx1, bby1, bbt2, bbx2, bby2 = rand_out(inputs.size(), (input_size)//2, length//2, cfg.mix_type)

        # Delete out
        zero_tensor = torch.zeros(inputs.size()).cuda()
        inputs[:, :, bbt1:bbt2, bbx1:bbx2, bby1:bby2] = zero_tensor[:, :, bbt1:bbt2, bbx1:bbx2, bby1:bby2]

    else:
        print('mixtype error')
        return

    return inputs, labels, lam, rand_index