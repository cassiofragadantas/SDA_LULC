### To implemente the Transformer Framework I used the code from this website : https://www.kaggle.com/code/arunmohan003/transformer-from-scratch-using-pytorch

import torch
import torch.nn as nn
import sys
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import sys
from sklearn.utils import shuffle
from model_pytorch import TempCNNDisentangleV4, SupervisedContrastiveLoss 
import time
from sklearn.metrics import f1_score, confusion_matrix
import torch.nn.functional as F


def sim_dist_specifc_loss_spc(spec_emb, ohe_label, ohe_dom, scl, epoch):
    norm_spec_emb = nn.functional.normalize(spec_emb)
    hash_label = {}
    new_combined_label = []
    for v1, v2 in zip(ohe_label, ohe_dom):
        key = "%d_%d"%(v1,v2)
        if key not in hash_label:
            hash_label[key] = len(hash_label)
        new_combined_label.append( hash_label[key] )
    new_combined_label = torch.tensor(np.array(new_combined_label), dtype=torch.int64)
    return scl(norm_spec_emb, new_combined_label, epoch=epoch)



def sim_dist_specifc_loss(spec_emb, class_label, domain_label):
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
    
def similar_specific_loss(spec_emb, class_label, domain_label):
    label_mask = np.matmul(class_label, np.transpose(class_label))
    label_mask = torch.tensor(label_mask).to(device)

    domain_mask = np.matmul(domain_label, np.transpose(domain_label))
    domain_mask = torch.tensor(domain_mask).to(device)

    mask = label_mask * domain_mask
    #mask = domain_mask


    L2_pointwise_dist = torch.cdist(spec_emb, spec_emb, p=2.0)
    L2_pointwise_dist = L2_pointwise_dist * mask
    upper_diag_L2_dist = torch.triu(L2_pointwise_dist, diagonal=1)
    upper_diag_mask = torch.triu(mask, diagonal=1)

    return torch.div( torch.sum(upper_diag_L2_dist), torch.sum(upper_diag_mask) )


def evaluation(model, dataloader, device):
    model.eval()
    tot_pred = []
    tot_labels = []
    for x_batch, y_batch in dataloader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        pred, _, _,_,_,_,_,_ = model(x_batch)
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
rng_seed = int(sys.argv[5]) if len(sys.argv) > 5 else 42

print(f'(Random seed set to {rng_seed})')
torch.manual_seed(rng_seed)
np.random.seed(rng_seed)

training_batch_size = 512#256#128

prefix_path = "DATA_%s/"%dataset

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
model = TempCNNDisentangleV4(n_classes).to(device)

learning_rate = 0.0001
loss_fn = nn.CrossEntropyLoss()
loss_fn_noReduction = nn.CrossEntropyLoss(reduction='none')
scl = SupervisedContrastiveLoss()

optimizer = torch.optim.AdamW(params=model.parameters(), lr=learning_rate)


epochs = 200
# Loop through the data
valid_f1 = 0.0
hashmap_cl_dom = {}
for i in range(n_classes):
    for j in range(2):
        k = "%d_%d"%(i,j)
        if k not in hashmap_cl_dom:
            hashmap_cl_dom[k] = len(hashmap_cl_dom)



for epoch in range(epochs):
    start = time.time()
    model.train()
    tot_loss = 0.0
    domain_loss = 0.0
    contra_tot_loss = 0.0
    den = 0
    #p = float(epoch) / epochs
    lambda_val = 1.0#(2. / (1. + np.exp(-10 * p))) - 1


    for x_batch, y_batch, dom_batch in train_dataloader:
        if x_batch.shape[0] != training_batch_size:
            continue

        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        dom_batch = dom_batch.to(device)
        optimizer.zero_grad()
        #pred, inv_emb, spec_emb_dc, spec_dc_pred = model(x_batch, alpha=lambda_val)
        pred, inv_emb, spec_emb_d, spec_d_pred, inv_emb_n1, spec_emb_n1, inv_fc_feat, spec_fc_feat = model(x_batch)

        ohe_label = F.one_hot(y_batch,num_classes=n_classes).cpu().detach().numpy()
        ohe_dom = F.one_hot(dom_batch,num_classes=2).cpu().detach().numpy()

        ##### DOMAIN CLASSIFICATION #####
        loss_ce_spec_dom = loss_fn(spec_d_pred, dom_batch)

        ##### MIXED MAINFOLD & CONTRASTIVE LEARNING ####
        
        cl_labels_npy = y_batch.cpu().detach().numpy()
        dummy_labels_npy = np.ones_like(cl_labels_npy) * n_classes
        y_mix_labels = np.concatenate([ cl_labels_npy , cl_labels_npy],axis=0)
        
        
        #DOMAIN LABEL FOR DOMAIN-CLASS SPECIFIC EMBEDDING and DOMAIN SPECIFIC EMBEDDING IS 0 OR 1 
        spec_dc_dom_labels = dom_batch.cpu().detach().numpy()
        #DOMAIN LABEL FOR INV EMBEDDING IS 2
        inv_dom_labels = np.ones_like(spec_dc_dom_labels) * 2

        dom_mix_labels = np.concatenate([inv_dom_labels, spec_dc_dom_labels],axis=0)
        joint_embedding = torch.concat([inv_emb, spec_emb_d])


        mixdl_loss_supContraLoss = sim_dist_specifc_loss_spc(joint_embedding, y_mix_labels, dom_mix_labels, scl, epoch)
        joint_embedding_n1 = torch.concat([inv_emb_n1, spec_emb_n1])
        mixdl_loss_supContraLoss_n1 = sim_dist_specifc_loss_spc(joint_embedding_n1, y_mix_labels, dom_mix_labels, scl, epoch)

        joint_embedding_fc_feat = torch.concat([inv_fc_feat, spec_fc_feat])
        mixdl_loss_supContraLoss_fc = sim_dist_specifc_loss_spc(joint_embedding_fc_feat, y_mix_labels, dom_mix_labels, scl, epoch)
        
        ####################################

        loss_cl_spec = loss_ce_spec_dom#loss_ce_spec_cdom #+loss_ce_spec_dom
        contra_loss = mixdl_loss_supContraLoss + mixdl_loss_supContraLoss_n1 + mixdl_loss_supContraLoss_fc
        loss = loss_fn(pred, y_batch) + contra_loss + loss_cl_spec #+ loss_ce_spec_dom  #+ dom_loss

        loss.backward() # backward pass: backpropagate the prediction loss
        optimizer.step() # gradient descent: adjust the parameters by the gradients collected in the backward pass
        tot_loss+= loss.cpu().detach().numpy()
        contra_tot_loss+= contra_loss.cpu().detach().numpy()
        den+=1.
    
    
    end = time.time()
    pred_valid, labels_valid = evaluation(model, valid_dataloader, device)
    f1_val = f1_score(labels_valid, pred_valid, average="weighted")
    if f1_val > valid_f1:
        torch.save(model.state_dict(), "model_REFeD_%s_%d_%d_%d.pth"%(dataset, source_year, id_, target_year))
        valid_f1 = f1_val
        pred_test, labels_test = evaluation(model, test_dataloader, device)
        f1 = f1_score(labels_test, pred_test, average="weighted")
        print("TOT AND CONTRA LOSS at Epoch %d: %.4f %.4f with acc on TEST TARGET SET %.2f with training time %d"%(epoch, tot_loss/den, contra_tot_loss/den, 100*f1, (end-start)))
        #print(confusion_matrix(labels_test, pred_test))
    else:
        print("TOT AND CONTRA AND TRAIN LOSS at Epoch %d: %.4f %.4f"%(epoch, tot_loss/den, contra_tot_loss/den))
    sys.stdout.flush()



