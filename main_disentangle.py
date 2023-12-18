
import torch
import torch.nn as nn
import sys
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import sys
from sklearn.utils import shuffle
from model_pytorch import TempCNNDisentangle
import time
from sklearn.metrics import f1_score
import torch.nn.functional as F

def sim_dist_specific_loss(spec_emb, class_label, domain_label):
    margin = 1.0
    label_mask = np.matmul(class_label, np.transpose(class_label))
    label_mask = torch.tensor(label_mask).to(device)

    domain_mask = np.matmul(domain_label, np.transpose(domain_label))
    domain_mask = torch.tensor(domain_mask).to(device)

    mask = label_mask * domain_mask
    neg_mask = 1 - mask

    L2_pointwise_dist = torch.cdist(spec_emb, spec_emb, p=2.0)
    pos_dist = mask * L2_pointwise_dist
    neg_dist = neg_mask * L2_pointwise_dist

    ud_pos_dist = torch.triu(pos_dist, diagonal=1)
    ud_neg_dist = torch.triu(neg_dist, diagonal=1)
    ud_pos_mask = torch.triu(mask, diagonal=1)
    ud_neg_mask = torch.triu(neg_mask, diagonal=1)

    pos_elem = torch.div( torch.sum(ud_pos_dist), torch.sum(ud_pos_mask) )
    neg_elem = torch.div( torch.sum(F.relu( margin - ud_neg_dist )), torch.sum(ud_neg_mask)  )
    return (pos_elem + neg_elem)/2

def evaluation(model, dataloader, device):
    model.eval()
    tot_pred = []
    tot_labels = []
    for x_batch, y_batch in dataloader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        pred, _, _,_, _, _ = model(x_batch)
        pred_npy = np.argmax(pred.cpu().detach().numpy(), axis=1)
        tot_pred.append( pred_npy )
        tot_labels.append( y_batch.cpu().detach().numpy())
    tot_pred = np.concatenate(tot_pred)
    tot_labels = np.concatenate(tot_labels)
    return tot_pred, tot_labels

source_year = int(sys.argv[1])
id_ = int(sys.argv[2])
target_year = int(sys.argv[3])
rng_seed = int(sys.argv[4]) if len(sys.argv) > 4 else 42

print(f'(Random seed set to {rng_seed})')
torch.manual_seed(rng_seed)
np.random.seed(rng_seed)

path_source = './DATA/' #f'./DATA_CVL_{source_year}/'
path_target = './DATA/' #f'./DATA_CVL_{target_year}/'

training_batch_size = 128

train_target_data = np.load("%strain_data_%d_%d.npy"%(path_target, id_, target_year))
train_target_label = np.load("%strain_label_%d_%d.npy"%(path_target, id_, target_year))

train_source_data = np.load("%sdata_%d.npy"%(path_source, source_year))
train_source_label = np.load("%sgt_data_%d.npy"%(path_source,source_year))
if train_source_label.ndim > 1 and train_source_label.shape[1] > 2:
    train_source_label = train_source_label[:,2]

train_data = np.concatenate([train_target_data, train_source_data],axis=0)
train_label = np.concatenate([train_target_label, train_source_label],axis=0)

train_domain_label = np.concatenate([np.zeros(train_target_label.shape[0]), np.ones(train_source_label.shape[0])], axis=0)

valid_data = np.load("%svalid_data_%d_%d.npy"%(path_target, id_,target_year))
valid_label = np.load("%svalid_label_%d_%d.npy"%(path_target, id_,target_year))

test_data = np.load("%stest_data_%d_%d.npy"%(path_target, id_, target_year))
test_label = np.load("%stest_label_%d_%d.npy"%(path_target, id_, target_year))

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
model = TempCNNDisentangle(train_data.shape[2], train_data.shape[1], n_classes).to(device)

learning_rate = 0.0001
loss_fn = nn.CrossEntropyLoss()
#loss_mse = nn.MSELoss()
loss_mae = nn.L1Loss()
optimizer = torch.optim.AdamW(params=model.parameters(), lr=learning_rate)

epochs = 200
# Loop through the data
valid_f1 = 0.0
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
        pred, inv_emb, spec_emb, reco, dom_pred, spec_dom_pred = model(x_batch)

        ohe_label = F.one_hot(y_batch,num_classes=n_classes).cpu().detach().numpy()
        ohe_dom = F.one_hot(dom_batch,num_classes=2).cpu().detach().numpy()

        dom_loss = loss_fn(dom_pred, dom_batch)
        sdl_loss = sim_dist_specific_loss(spec_emb, ohe_label, ohe_dom)
        loss = loss_fn(pred, y_batch) + loss_mae(reco, x_batch) + sdl_loss + dom_loss

        loss.backward() # backward pass: backpropagate the prediction loss
        optimizer.step() # gradient descent: adjust the parameters by the gradients collected in the backward pass
        tot_loss+= loss.cpu().detach().numpy()
        domain_loss+=dom_loss
        contra_tot_loss+=sdl_loss
        den+=1.

    end = time.time()
    pred_valid, labels_valid = evaluation(model, valid_dataloader, device)
    f1_val = f1_score(labels_valid, pred_valid, average="weighted")
    if f1_val > valid_f1:
        #torch.save(model.state_dict(), "model_combined_source_target_wreco_%d_%d_%d.pth"%(source_year, id_, target_year))
        torch.save(model.state_dict(), "model_combined_source_target_dis_%d_%d_%d.pth"%(source_year, id_, target_year))
        valid_f1 = f1_val
        pred_test, labels_test = evaluation(model, test_dataloader, device)
        f1 = f1_score(labels_test, pred_test, average="weighted")
        print("TOT AND DOMAIN AND CONTRA LOSS at Epoch %d: %.4f %.4f %.4f with acc on TEST TARGET SET %.2f with training time %d"%(epoch, tot_loss/den, contra_tot_loss/den, domain_loss/den, 100*f1,(end-start)))
    else:
        print("TOT AND DOMAIN AND CONTRA AND TRAIN LOSS at Epoch %d: %.4f %.4f %.4f"%(epoch, tot_loss/den, contra_tot_loss/den, domain_loss/den))
    sys.stdout.flush()