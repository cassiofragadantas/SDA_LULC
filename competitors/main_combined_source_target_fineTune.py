### To implemente the Transformer Framework I used the code from this website : https://www.kaggle.com/code/arunmohan003/transformer-from-scratch-using-pytorch

import torch
import torch.nn as nn
import sys
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import sys
from sklearn.utils import shuffle
#from model_transformer import TransformerEncoder
from model_pytorch import TempCNN#, Inception
import time
from sklearn.metrics import f1_score

def evaluation(model, dataloader, device):
    model.eval()
    tot_pred = []
    tot_labels = []
    for x_batch, y_batch in dataloader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        pred, _ = model(x_batch)
        pred_npy = np.argmax(pred.cpu().detach().numpy(), axis=1)
        tot_pred.append( pred_npy )
        tot_labels.append( y_batch.cpu().detach().numpy())
    tot_pred = np.concatenate(tot_pred)
    tot_labels = np.concatenate(tot_labels)
    return tot_pred, tot_labels


source_year = int(sys.argv[1])
id_ = int(sys.argv[2])
target_year = int(sys.argv[3])
dataset = sys.argv[4]
prefix_path = "data_%s/"%dataset

train_target_data = np.load(prefix_path+"train_data_%d_%d.npy"%(id_, target_year))
train_target_label = np.load(prefix_path+"train_label_%d_%d.npy"%(id_, target_year))

train_source_data = np.load(prefix_path+"data_%d.npy"%(source_year))
train_source_label = np.load(prefix_path+"gt_data_%d.npy"%source_year)#[:,2]

if len( train_source_label.shape) == 2:
    train_source_label = train_source_label[:,2]

if len( train_target_label.shape) == 2:
    train_target_label = train_target_label[:,2]

#train_data = np.concatenate([train_target_data, train_source_data],axis=0)
#train_label = np.concatenate([train_target_label, train_source_label],axis=0)

valid_data = np.load(prefix_path+"valid_data_%d_%d.npy"%(id_,target_year)) 
valid_label = np.load(prefix_path+"valid_label_%d_%d.npy"%(id_,target_year))

test_data = np.load(prefix_path+"test_data_%d_%d.npy"%(id_, target_year))
test_label = np.load(prefix_path+"test_label_%d_%d.npy"%(id_, target_year))

n_classes = len( np.unique(train_source_label))

train_source_label = train_source_label - 1
train_target_label = train_target_label - 1
test_label = test_label - 1
valid_label = valid_label - 1

x_train_source = torch.tensor(train_source_data, dtype=torch.float32)
y_train_source = torch.tensor(train_source_label, dtype=torch.int64)

x_train_target = torch.tensor(train_target_data, dtype=torch.float32)
y_train_target = torch.tensor(train_target_label, dtype=torch.int64)

x_valid = torch.tensor(valid_data, dtype=torch.float32)
y_valid = torch.tensor(valid_label, dtype=torch.int64)

x_test = torch.tensor(test_data, dtype=torch.float32)
y_test = torch.tensor(test_label, dtype=torch.int64)

train_dataset_source = TensorDataset(x_train_source, y_train_source)
train_dataset_target = TensorDataset(x_train_target, y_train_target)
valid_dataset = TensorDataset(x_valid, y_valid)
test_dataset = TensorDataset(x_test, y_test)



train_source_dataloader = DataLoader(train_dataset_source, shuffle=True, batch_size=256)
train_target_dataloader = DataLoader(train_dataset_target, shuffle=True, batch_size=256)
valid_dataloader = DataLoader(valid_dataset, shuffle=False, batch_size=2048)
test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=2048)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

seq_length = train_source_data.shape[2]
embedding_size = 64
#model = Inception(n_classes).to(device)
model = TempCNN(n_classes).to(device)


learning_rate = 0.0001
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(params=model.parameters(), lr=learning_rate)




epochs = 200

# PRE-TRAINING ON SOURCE DATA
for epoch in range(epochs//2):
    print("EPOCH PRETRAINING %d"%epoch)
    for x_batch, y_batch in train_source_dataloader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        optimizer.zero_grad()
        pred, _ = model(x_batch)
        loss = loss_fn(pred, y_batch)
        loss.backward() # backward pass: backpropagate the prediction loss
        optimizer.step() # gradient descent: adjust the parameters by the gradients collected in the backward pass

# FINE-TUNING ON TARGET DATA
valid_f1 = 0.0
for epoch in range(epochs//2):
    start = time.time()
    model.train()
    tot_loss = 0.0
    den = 0
    for x_batch, y_batch in train_target_dataloader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        optimizer.zero_grad()
        pred, _ = model(x_batch)
        loss = loss_fn(pred, y_batch)

        loss.backward() # backward pass: backpropagate the prediction loss
        optimizer.step() # gradient descent: adjust the parameters by the gradients collected in the backward pass
        tot_loss+= loss.cpu().detach().numpy()
        den+=1.
    
    end = time.time()
    pred_valid, labels_valid = evaluation(model, valid_dataloader, device)
    f1_val = f1_score(labels_valid, pred_valid, average="weighted")
    if f1_val > valid_f1:
        torch.save(model.state_dict(), "model_combined_source_target_fineTuned_%s_%d_%d_%d.pth"%(dataset, source_year, id_, target_year))
        valid_f1 = f1_val
        pred_test, labels_test = evaluation(model, test_dataloader, device)
        f1 = f1_score(labels_test, pred_test, average="weighted")
        print("TRAIN LOSS at Epoch %d: %.4f with acc on TEST TARGET SET %.2f with training time %d"%(epoch, tot_loss/den, 100*f1,(end-start)))
    else:
        print("TRAIN LOSS at Epoch %d: %.4f"%(epoch, tot_loss/den))
    sys.stdout.flush()



