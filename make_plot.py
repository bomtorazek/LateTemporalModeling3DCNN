import matplotlib.pyplot as plt
from os.path import join

PATH = '.checkpoint/cvpr_rgb_r2plus1d_64f_34_bert10_split0'
opt_cnt = 1 if '64f' in PATH else 0  # additional print cause 64f should have half batch

def argmax(lst):
  return lst.index(max(lst))

with open(join(PATH, 'log.txt')) as f:
    data = f.readlines()
    for idx, row in enumerate(data):
        if '[' in row:
            break
    epoch = []
    train_acc = []
    val_acc = []

    cnt = 0
    while 1:
        data_list1 = data[idx+1+opt_cnt].split()
        print(data_list1, idx)
        epoch.append(int(data_list1[2]))
        train_acc.append(float(data_list1[4]))
        val_acc.append(float(data[idx+3+opt_cnt].split()[3]))
        if (idx + 6 + opt_cnt> len(data)):
            break
        else:
            if 'kaydedildi' in data[idx+5+opt_cnt] or 'kaydedildi' in data[idx+6+opt_cnt]:
                cnt+=1
            if  'learning' in data[idx+5+opt_cnt] or 'learning' in data[idx+6+opt_cnt]:
                cnt+=1
            idx += 5 + cnt +opt_cnt
            cnt =0
    t_idx = argmax(train_acc)
    v_idx = argmax(val_acc)
    plt.plot(epoch, train_acc, label = 'train')
    plt.plot(epoch, val_acc, label = 'val')
    
    plt.text(len(epoch)/2.5, 82.5, f"Best Training Acc:{train_acc[t_idx]}% @ Epoch:{epoch[t_idx]} \nBest Val Acc:{val_acc[v_idx]}% @ Epoch:{epoch[v_idx]}", bbox=dict(facecolor='red', alpha=0.5))
    plt.ylim(80,100)
    plt.grid()
    plt.legend()
    plt.title("Train, Val Accuracy - Epoch")
    plt.savefig(join(PATH, 'training_log.png'))

