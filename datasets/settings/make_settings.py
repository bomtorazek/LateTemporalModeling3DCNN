
from os.path import join
from os import listdir
import csv

dataset= 'cvpr'
frame_path = '../cvpr_frames/'
train_lbl_path = '/home/esuh/data/cvpr/Track2.1/ARID1.1_t1_train_pub.csv'
val_lbl_path = '/home/esuh/data/cvpr/Track2.1/ARID1.1_t1_validation_gt_pub.csv'
#label mapping
# Run:5, Sit:6, Stand:7, Turn:8, Walk:9, wave:10, 
#     0,     1,       2,      3,      4,       5, -5

ano_dict = {}
with open(val_lbl_path, 'r') as an:
    reader = csv.reader(an)
    next(reader)
    for row in reader:
        try:
            ano_dict[row[1].split('/')[1][:-4]] = int(row[2])-5
        except:
            ano_dict[row[1][:-4]] = int(row[2])-5
# validation frames도 만들어야 한다. 같은 폴더에 넣고 settings로 구분하는 것 같다. 완료

# 50_FIRST_DATES_kick_f_cm_np1_ba_med_19 47 20
with open(f"{dataset}/val_rgb_split0.txt", "w") as f:
    for folder in listdir(frame_path):

        if folder in ano_dict.keys():
            print(folder)
            frames = 0
            for jpg in listdir(join(frame_path,folder)):
                if 'img' in jpg:
                    frames+=1
            lbl = ano_dict[folder]
            f.write(f'{folder} {frames} {lbl}\n')

