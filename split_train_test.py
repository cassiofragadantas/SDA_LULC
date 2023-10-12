import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
import sys
from sklearn.utils import shuffle

def getDataIdx(objs, groups, orig_data, labels):
    data_idx = []
    for obj_id in objs:
        idx = np.where(groups == obj_id)[0]
        data_idx.append( idx )
    return np.concatenate(data_idx,axis=0)

year = int(sys.argv[1])
gt = np.load("gt_data_%d.npy"%year)
data = np.load("data_%d.npy"%year)

train_perc = .5
valid_perc = .2
n_repeated_hold_out = 5

for i in range(n_repeated_hold_out):
    print("hold out %d"%i)
    data, gt = shuffle(data, gt)

    groups = gt[:,3]
    labels = gt[:,2]
    lab2objs = {}
    train_objs_ids = []
    valid_objs_ids = []
    test_objs_ids = []
    for l in np.unique( labels ):
        idx = np.where(labels == l)[0]
        obj_idx = shuffle( np.unique( groups[idx] ), random_state=i*100 )
        n_objs = len(obj_idx)
        limit_train = int(train_perc * n_objs)
        limit_valid = int((train_perc+valid_perc) * n_objs)
        train_objs_ids.append( obj_idx[0:limit_train] )
        valid_objs_ids.append( obj_idx[limit_train:limit_valid])
        test_objs_ids.append( obj_idx[limit_valid::] )

    train_objs_ids = np.concatenate(train_objs_ids)
    valid_objs_ids = np.concatenate(valid_objs_ids)
    test_objs_ids = np.concatenate(test_objs_ids)

    train_objs_ids = np.unique(train_objs_ids)
    valid_objs_ids = np.unique(valid_objs_ids)
    test_objs_ids = np.unique(test_objs_ids)

    train_idx = getDataIdx(train_objs_ids, groups, data, labels)
    valid_idx = getDataIdx(valid_objs_ids, groups, data, labels)
    test_idx = getDataIdx(test_objs_ids, groups, data, labels)

    np.save("train_data_%d_%d.npy"%(i,year), data[train_idx])
    np.save("train_label_%d_%d.npy"%(i,year), labels[train_idx])

    np.save("valid_data_%d_%d.npy"%(i,year), data[valid_idx])
    np.save("valid_label_%d_%d.npy"%(i,year), labels[valid_idx])

    np.save("test_data_%d_%d.npy"%(i,year), data[test_idx])
    np.save("test_label_%d_%d.npy"%(i,year), labels[test_idx])