
from os.path import join
from os import listdir
import csv

modality = 'rgb' # rgb or flow
dataset= 'semi_cvpr' # or cvpr_le for light-enhanced data (not implemented yet)
frame_path = '../semi_cvpr_frames/' # or cvpr_le_frames
val_lbl_path = '/home/esuh/data/cvpr/Track2.2/for_label'
test_lbl_path = '' #FIXME after 4/28
lbl_dict = {'drink': 0,  'jump': 1,  'pick':2,  'pour':3,  'push':4}
cls_dict = {}

## Make a dictionary from the annotation file
for folder in listdir(val_lbl_path):
    for clip in listdir(join(val_lbl_path, folder)):
        name = 'validation_'+clip[:-4]
        cls_dict[name] = lbl_dict[folder]

## make {}_{}_split{}.txt
# 50_FIRST_DATES_kick_f_cm_np1_ba_med_19 47 20
with open(f"{dataset}/l_train_{modality}_split0.txt", "w") as lf:
    with open(f"{dataset}/u_train_{modality}_split0.txt", "w") as uf:
        with open(f"{dataset}/val_{modality}_split0.txt", "w") as vf:
            for folder in listdir(frame_path):
                if 'train_' in folder[:6]:
                    frames = 0
                    for jpg in listdir(join(frame_path,folder)):
                        if 'img' in jpg:
                            frames+=1
                    uf.write(f'{folder} {frames}\n')
                elif 'validation' in folder:
                    frames = 0
                    for jpg in listdir(join(frame_path,folder)):
                        if 'img' in jpg:
                            frames+=1
                    vf.write(f'{folder} {frames} {cls_dict[folder]}\n')
                else:
                    for cls in lbl_dict.keys():
                        if f'_{cls}_' in folder:
                            frames = 0
                            for jpg in listdir(join(frame_path,folder)):
                                if 'img' in jpg:
                                    frames+=1
                            lf.write(f'{folder} {frames} {lbl_dict[cls]}\n')


