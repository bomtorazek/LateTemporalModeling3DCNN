from os.path import join
from os import listdir
from collections import defaultdict


FOLD = 2

train_path = './cvpr/train_rgb_split0.txt'
val_path = './cvpr/val_rgb_split0.txt'
inst_dict = defaultdict(list)
with open(train_path, 'r') as f:
    data = f.readlines()
    cnt = 0
    for row in data:
        cnt+=1
        lbl = row.split()[2]
        frames = row.split()[1]
        clip = row.split()[0]
        instance = clip.split('_')[0]+'_'+clip.split('_')[1]
        inst_dict[instance].append([clip, frames, lbl])


# with open(path2, 'r') as f:
#     data = f.readlines()
#     for row in data:
#         lbl = row.split()[2]
#         frames = row.split()[1]
#         clip = row.split()[0]
#         lbl_dict[int(lbl)].append([frames,clip])



for fold in range(1,FOLD+1):
    t_cnt = [0]*6
    v_cnt = [0]*6
    train_rem = [(i+fold)%FOLD for i in range(0,FOLD-1) ]
    with open(f'./cvpr/train_rgb_split{FOLD}{fold}.txt', 'w') as tf:
        with open(f'./cvpr/val_rgb_split{FOLD}{fold}.txt', 'w') as vf:
            for idx, inst in enumerate(inst_dict.keys()): # instance
                if idx%FOLD in train_rem:
                    for clip in inst_dict[inst]: # clip
                        tf.write(f'{clip[0]} {clip[1]} {clip[2]}\n')
                        # t_cnt[int(clip[2])]+=1
                else:
                    for clip in inst_dict[inst]: # clip
                        # v_cnt[int(clip[2])] +=1
                        vf.write(f'{clip[0]} {clip[1]} {clip[2]}\n')

    # print([i/sum(t_cnt) for i in t_cnt])
    # print([(i+j)/(sum(t_cnt)+sum(v_cnt)) for i, j in zip(t_cnt, v_cnt)])
                                
                    
                