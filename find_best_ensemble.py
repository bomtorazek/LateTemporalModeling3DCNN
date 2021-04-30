import argparse
import pandas as pd
import copy
import numpy as np
import csv

from os import listdir
from os.path import isfile, join
from itertools import combinations
from sklearn.metrics import accuracy_score

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv-folder', type=str, default='ensemble_models')
    parser.add_argument('--gt-path', type=str, default='/home/esuh/LateTemporalModeling3DCNN/datasets/settings/semi_cvpr/test_rgb_split111.txt')
    parser.add_argument('--out-path', type=str, default='./ensembled_track2_pred.csv')
    
    
   

    args = parser.parse_args()
    return args



def make_gt(gt_path):
    gt = []
    with open(gt_path) as file:
        for row in file.readlines():
            spt = row.split()
            gt.append([int(spt[0]), int(spt[2])]) # FIXME
    gt = [i[1] for i in sorted(gt)]
    return gt
        


def average_ensemble(tbe):
    averaged_probs = np.zeros((tbe[0].shape[0], tbe[0].shape[1]-2))
    for prob_csv in tbe:
        averaged_probs = averaged_probs + prob_csv.values[:,2:]
    averaged_probs /= len(tbe)
    assert averaged_probs.any() <= 1.0
    averaged_preds = np.argmax(averaged_probs,axis = 1)
    return averaged_preds

def max_ensemble(tbe):
    max_preds = np.zeros((tbe[0].shape[0], 2))
    for prob_csv in tbe:
        only_prob = prob_csv.values[:,2:]
        for i in range(max_preds.shape[0]):
            if np.amax(only_prob[i,:]) > max_preds[i,0]:
                max_preds[i,0]=np.amax(only_prob[i,:])
                max_preds[i,1] = np.argmax(only_prob[i,:])
    max_preds = list(max_preds[:,1].astype(np.int32))
    return max_preds

def vote_ensemble(tbe): 
    voted_preds = []
    for prob_csv in tbe:
        only_prob = prob_csv.values[:,2:]
        only_pred = np.argmax(only_prob, axis = 1)
        voted_preds.append(np.expand_dims(only_pred,axis=1))
    voted_preds = np.concatenate(voted_preds,axis=1)

    voted_pd = []
    for i in range(voted_preds.shape[0]):
        voted_pd.append(np.bincount(voted_preds[i,:]).argmax())
    
    return voted_pd


def make_combinations(n): 
    # if n = 3, [[0], [1], [2], [0,1], [0,2], [1,2], [0,1,2]]
    combinations_list = []
    for x in range(1,n+1):
        combinations_list += list(combinations(range(n),x))

    return combinations_list 
    

def do_ensemble(args, gt):

    folder = args.csv_folder
    probs_list = []
    model_list = []
    file_list = []
    for f in listdir(folder):
        if isfile(join(folder,f)):
            model_list.append(f) # file name
            probs_list.append(pd.read_csv(join(folder,f))) #[ probs[0], probs[1], ...]
            file_list.append(f)

    num_models = len(probs_list)
    comb_list = make_combinations(num_models)

    ensemble_method = None
    ensemble_indices = None
    best_acc = 0
    best_pred = None
    for comb in comb_list: 
        print( "----------{} out of {} models are being ensembled...----------".format(comb,num_models))
        to_be_ensembled = [probs_list[idx] for idx in comb] # probabilities
        # averaging
        averaged_preds = average_ensemble(to_be_ensembled)
        avg_acc = accuracy_score(gt, averaged_preds)
        # maximizing
        maximized_preds = max_ensemble(to_be_ensembled)
        maxed_acc = accuracy_score(gt, maximized_preds)
        # voting
        voted_preds = vote_ensemble(to_be_ensembled)
        vot_acc = accuracy_score(gt, voted_preds)

        print(f"accuracies: avg{avg_acc*100}, max{maxed_acc*100}, voting{vot_acc*100}")
        
        if best_acc < avg_acc:
            best_acc = avg_acc
            ensemble_method = 'avg'
            ensemble_indices = comb
            best_pred = averaged_preds
        if best_acc < maxed_acc:
            best_acc = maxed_acc
            ensemble_method = 'max'
            ensemble_indices = comb
            best_pred = maximized_preds
        if best_acc < vot_acc:
            best_acc = vot_acc
            ensemble_method = 'vote'
            ensemble_indices = comb
            best_pred = voted_preds

    return best_acc, ensemble_method, ensemble_indices, best_pred, file_list


        





if __name__ == '__main__':
    args = get_args()
    gt = make_gt(args.gt_path)
    best_acc, ensemble_method, ensemble_indices, best_pred, file_list = do_ensemble(args,gt)
    print(f"best_acc = {best_acc*100}, method = {ensemble_method}, indices = {ensemble_indices}")
    for file in file_list:
        print(file)
    with open(args.out_path, 'w') as f:
        pencil = csv.writer(f) 
        pencil.writerow(['VideoID', 'Video', 'ClassID'])
        for idx, pred in enumerate(best_pred):
            pencil.writerow([idx, str(idx)+'.mp4', pred]) 
            # FIXME assume that the orders of gt and pred are the same
            # and the number of clip is the same with idx

