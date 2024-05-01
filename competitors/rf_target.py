#import torch
#import torch.nn as nn
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

#source_year = int(sys.argv[1])
id_ = int(sys.argv[1])
target_year = int(sys.argv[2])
dataset = sys.argv[3] if len(sys.argv) > 3 else 'Koumbia'
rng_seed = int(sys.argv[4]) if len(sys.argv) > 4 else 42

print(f'(Random seed set to {rng_seed})')
np.random.seed(rng_seed)

path_target = f'../DATA_{dataset}/'

train_data = np.load("%strain_data_%d_%d.npy"%(path_target, id_, target_year) )
train_label = np.load("%strain_label_%d_%d.npy"%(path_target, id_, target_year))

valid_data = np.load("%svalid_data_%d_%d.npy"%(path_target, id_,target_year)) 
valid_label = np.load("%svalid_label_%d_%d.npy"%(path_target, id_,target_year))

test_data = np.load("%stest_data_%d_%d.npy"%(path_target, id_, target_year))
test_label = np.load("%stest_label_%d_%d.npy"%(path_target, id_,target_year))


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
    print('Training model with %d estimators...'%n_estim)
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
fileName = "rf_model_full_target_prediction_%s_%s"%(target_year,id_)
np.save(fileName, pred)
