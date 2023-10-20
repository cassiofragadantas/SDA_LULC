#import torch
#import torch.nn as nn
import os
#from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import sys
from sklearn.utils import shuffle
#from model_transformer import TransformerEncoder
#from model_pytorch import TempCNN, Inception
import time
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

source_year = int(sys.argv[1])
id_ = int(sys.argv[2])
target_year = int(sys.argv[3])
rng_seed = int(sys.argv[4]) if len(sys.argv) > 4 else 42

print(f'(Random seed set to {rng_seed})')
np.random.seed(rng_seed)

fileName = "rf_model_combined_prediction_%s_%s_%s"%(source_year,target_year,id_)
if os.path.exists(fileName+".npy"):
    exit(0)


train_data_source = np.load("./DATA/data_%d.npy"%source_year)
train_label_source = np.load("./DATA/gt_data_%d.npy"%source_year)[:,2]

train_data_target = np.load("./DATA/train_data_%d_%d.npy"%(id_,target_year)) 
train_label_target = np.load("./DATA/train_label_%d_%d.npy"%(id_,target_year))

train_data = np.concatenate([train_data_source, train_data_target],axis=0)
train_label = np.concatenate([train_label_source, train_label_target],axis=0)

valid_data = np.load("./DATA/valid_data_%d_%d.npy"%(id_,target_year)) 
valid_label = np.load("./DATA/valid_label_%d_%d.npy"%(id_,target_year))

test_data = np.load("./DATA/test_data_%d_%d.npy"%(id_,target_year)) 
test_label = np.load("./DATA/test_label_%d_%d.npy"%(id_,target_year))


nrow, nt, nb = train_data.shape
train_data = np.reshape(train_data, (nrow, nt * nb) )
nrow, nt, nb = valid_data.shape
valid_data = np.reshape(valid_data, (nrow, nt * nb) )
nrow, nt, nb = test_data.shape
test_data = np.reshape(test_data, (nrow, nt * nb) )

fval_previous = 0
final_estim = -1
for i in range(5):
    n_estim = (i+1)*100
    rf = RandomForestClassifier(n_estimators=n_estim)
    rf.fit(train_data,train_label)
    pred_valid = rf.predict(valid_data)
    fval = f1_score(valid_label,pred_valid, average="weighted")
    print("\t fval %.3f"%(fval*100) )
    if fval > fval_previous:
        fval_previous = fval
        final_estim = n_estim

new_data = np.concatenate( [train_data, valid_data], axis=0)
new_label = np.concatenate( [train_label, valid_label], axis=0)

rf = RandomForestClassifier(n_estimators=final_estim)
rf.fit(new_data, new_label)
pred = rf.predict(test_data)
np.save(fileName, pred)