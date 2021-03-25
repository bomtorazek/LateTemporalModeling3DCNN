
from os.path import join
from os import listdir
import csv

trainvaltest = 'val' #  train or val or test
modality = 'rgb' # rgb or flow
dataset= 'cvpr' # or cvpr_le for light-enhanced data (not implemented yet)
frame_path = '../cvpr_frames/' # or cvpr_le_frames
train_lbl_path = '../ARID1.1_t1_train_pub.csv'
val_lbl_path = '../ARID1.1_t1_validation_gt_pub.csv'
test_lbl_path = '' #FIXME after 4/28
lbl_path = {'train': train_lbl_path,'val': val_lbl_path, 'test': test_lbl_path}


## label mapping
# Run:5, Sit:6, Stand:7, Turn:8, Walk:9, wave:10, 
#     0,     1,       2,      3,      4,       5, -5씩 하면 됩니다.


## Make a dictionary from the annotation file
ano_dict = {}
with open(lbl_path[trainvaltest], 'r') as an:
    reader = csv.reader(an)
    next(reader)
    for row in reader:
        if trainvaltest == 'train':
            ano_dict[row[1].split('/')[1][:-4]] = int(row[2])-5
        else:
            ano_dict[row[1][:-4]] = int(row[2])-5

## make {}_{}_split{}.txt
# 50_FIRST_DATES_kick_f_cm_np1_ba_med_19 47 20
# TODO 5-fold split 
with open(f"{dataset}/{trainvaltest}_{modality}_split0.txt", "w") as f:
    for folder in listdir(frame_path):

        if folder in ano_dict.keys():
            print(folder)
            frames = 0
            for jpg in listdir(join(frame_path,folder)):
                if 'img' in jpg:
                    frames+=1
            lbl = ano_dict[folder]
            f.write(f'{folder} {frames} {lbl}\n')

