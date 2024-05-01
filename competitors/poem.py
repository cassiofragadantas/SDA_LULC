### To implemente the Transformer Framework I used the code from this website : https://www.kaggle.com/code/arunmohan003/transformer-from-scratch-using-pytorch

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import sys
sys.path.append('..')
from sklearn.utils import shuffle
#from model_transformer import TransformerEncoder
from model_pytorch import TempCNNPoem
import time
from sklearn.metrics import f1_score, confusion_matrix
import torch.nn.functional as F
import warnings
warnings.filterwarnings('ignore')
from copy import deepcopy
from collections import OrderedDict
#torch.set_default_dtype(torch.float16)


def evaluation(model, dataloader, device):
    model.eval()
    tot_pred = []
    tot_labels = []
    for x_batch, y_batch in dataloader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        pred, _, _, _, _ = model(x_batch)
        pred_npy = np.argmax(pred.cpu().detach().numpy(), axis=1)
        tot_pred.append( pred_npy )
        tot_labels.append( y_batch.cpu().detach().numpy())
    tot_pred = np.concatenate(tot_pred)
    tot_labels = np.concatenate(tot_labels)
    return tot_pred, tot_labels

source_year = int(sys.argv[1])
id_ = int(sys.argv[2])
target_year = int(sys.argv[3])
dataset = sys.argv[4] if len(sys.argv) > 4 else 'Koumbia'
rng_seed = int(sys.argv[5]) if len(sys.argv) > 5 else 42

training_batch_size = 512#256#128
#training_batch_size = 3

#prefix_path = "data_CVL3/"
#CVL3
#koumbia
prefix_path = "../DATA_%s/"%dataset

train_target_data = np.load(prefix_path+"train_data_%d_%d.npy"%(id_, target_year))
train_target_label = np.load(prefix_path+"train_label_%d_%d.npy"%(id_, target_year))

train_source_data = np.load(prefix_path+"data_%d.npy"%(source_year))
train_source_label = np.load(prefix_path+"gt_data_%d.npy"%source_year)#[:,2]

if len( train_source_label.shape) == 2:
    train_source_label = train_source_label[:,2]


train_data = np.concatenate([train_target_data, train_source_data],axis=0)
train_label = np.concatenate([train_target_label, train_source_label],axis=0)

train_domain_label = np.concatenate([np.zeros(train_target_label.shape[0]), np.ones(train_source_label.shape[0])], axis=0)

valid_data = np.load(prefix_path+"valid_data_%d_%d.npy"%(id_,target_year)) 
valid_label = np.load(prefix_path+"valid_label_%d_%d.npy"%(id_,target_year))

test_data = np.load(prefix_path+"test_data_%d_%d.npy"%(id_, target_year))
test_label = np.load(prefix_path+"test_label_%d_%d.npy"%(id_, target_year))

n_classes = len( np.unique(train_label))

train_label = train_label - 1
test_label = test_label - 1
valid_label = valid_label - 1

x_train = torch.tensor(train_data, dtype=torch.float32)
y_train = torch.tensor(train_label, dtype=torch.int64)
dom_train = torch.tensor(train_domain_label, dtype=torch.int64)

x_valid = torch.tensor(valid_data, dtype=torch.float32)
y_valid = torch.tensor(valid_label, dtype=torch.int64)

x_test = torch.tensor(test_data, dtype=torch.float32)
y_test = torch.tensor(test_label, dtype=torch.int64)

train_dataset = TensorDataset(x_train, y_train, dom_train)
valid_dataset = TensorDataset(x_valid, y_valid)
test_dataset = TensorDataset(x_test, y_test)

train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=training_batch_size)
valid_dataloader = DataLoader(valid_dataset, shuffle=False, batch_size=1024)
test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=1024)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
seq_length = train_data.shape[2]
model = TempCNNPoem(n_classes).to(device)

learning_rate = 0.0001
loss_fn = nn.CrossEntropyLoss()
loss_fn_noReduction = nn.CrossEntropyLoss(reduction='none')

optimizer = torch.optim.AdamW(params=model.parameters(), lr=learning_rate)

topK = 5
epochs = 200
# Loop through the data
valid_f1 = 0.0
hashmap_cl_dom = {}
for i in range(n_classes):
    for j in range(2):
        k = "%d_%d"%(i,j)
        if k not in hashmap_cl_dom:
            hashmap_cl_dom[k] = len(hashmap_cl_dom)


tot_num_iter = 0
queue_w_val = []
queue_f_val = []

for epoch in range(epochs):
    start = time.time()
    model.train()
    tot_loss = 0.0
    domain_loss = 0.0
    contra_tot_loss = 0.0
    den = 0

    for x_batch, y_batch, dom_batch in train_dataloader:
        if x_batch.shape[0] != training_batch_size:
            continue

        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        dom_batch = dom_batch.to(device)
        optimizer.zero_grad()
        #pred, inv_emb, spec_emb_dc, spec_dc_pred = model(x_batch, alpha=lambda_val)
        pred, pred_dom, pred_enc, inv_emb, spec_emb = model(x_batch)
        ohe_dom = F.one_hot(dom_batch,num_classes=2).cpu().detach().numpy()

        enc_batch_label = torch.tensor( np.concatenate( [np.zeros(ohe_dom.shape[0]), np.ones(ohe_dom.shape[0])], axis=0 ),dtype=torch.int64).to(device)

        loss_dom = loss_fn(pred_dom, dom_batch)
        loss_enc = loss_fn(pred_enc, enc_batch_label)
        loss_c = loss_fn(pred, y_batch)
        loss_s = torch.mean( torch.sum( nn.functional.normalize(spec_emb) * nn.functional.normalize(inv_emb), dim=1) )


        loss = loss_c +  loss_dom + loss_enc + loss_s
        loss.backward() # backward pass: backpropagate the prediction loss
        optimizer.step() # gradient descent: adjust the parameters by the gradients collected in the backward pass
        tot_loss+= loss.cpu().detach().numpy()
        den+=1.
        tot_num_iter+=1
    
    end = time.time()
    pred_valid, labels_valid = evaluation(model, valid_dataloader, device)
    f1_val = f1_score(labels_valid, pred_valid, average="weighted")
    


    if f1_val > valid_f1:
        torch.save(model.state_dict(), "poem_%s_%d_%d_%d.pth"%(dataset, source_year, id_, target_year))
        valid_f1 = f1_val
        pred_test, labels_test = evaluation(model, test_dataloader, device)
        f1 = f1_score(labels_test, pred_test, average="weighted")
        print("TOT LOSS at Epoch %d: %.4f with acc on TEST TARGET SET %.2f with training time %d"%(epoch, tot_loss/den, 100*f1, (end-start)))
    else:
        print("TOT LOSS at Epoch %d: %.4f"%(epoch, tot_loss/den))

    sys.stdout.flush()



