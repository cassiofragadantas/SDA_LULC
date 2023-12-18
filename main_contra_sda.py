import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import sys
import math
from sklearn.utils import shuffle
import time
#from model import TempCNN
from sklearn.metrics import f1_score, confusion_matrix
from model_pytorch import TempCNN, TempCNNWP

#from model_transformer import TransformerEncoder
from torch.optim.lr_scheduler import CosineAnnealingLR, CyclicLR, ExponentialLR, StepLR

def evaluation(model, dataloader, device):
    model.eval()
    tot_pred = []
    tot_labels = []
    for x_batch, y_batch in dataloader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        pred, _, _, _ = model(x_batch)
        pred_npy = np.argmax(pred.cpu().detach().numpy(), axis=1)
        tot_pred.append( pred_npy )
        tot_labels.append( y_batch.cpu().detach().numpy())
    tot_pred = np.concatenate(tot_pred)
    tot_labels = np.concatenate(tot_labels)
    return tot_pred, tot_labels


def rebalance_data(n_elems, X_train_target, Y_train_target):
    X_train_target, Y_train_target = shuffle(X_train_target, Y_train_target)
    to_add = n_elems - X_train_target.shape[0]
    X_train_target = np.concatenate([X_train_target, X_train_target[0:to_add]],axis=0)
    Y_train_target = np.concatenate([Y_train_target, Y_train_target[0:to_add]],axis=0)
    return X_train_target, Y_train_target


def createDataLoader(x, y, shuffling=False, batch_size = 1024):
    x_t = torch.tensor(x, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.int64)
    dataset = TensorDataset(x_t, y_t)
    dataloader = DataLoader(dataset, shuffle=shuffling, batch_size=batch_size)
    return dataloader

def createDataLoaderDouble(x_source, x_target, y_source, y_target, batch_size = 128):
    x_t = torch.tensor(x_target, dtype=torch.float32)
    y_t = torch.tensor(y_target, dtype=torch.int64)
    x_s = torch.tensor(x_source, dtype=torch.float32)
    y_s = torch.tensor(y_source, dtype=torch.int64)

    dataset = TensorDataset(x_s, y_s, x_t, y_t)
    dataloader = DataLoader(dataset, shuffle=True, batch_size=batch_size)
    return dataloader
    
def simpleDist(emb_source, emb_target, ohe_s, ohe_t, epoch, device ):
    margin = .6
    whole_emb = torch.cat([emb_source, emb_target],axis=0)
    whole_labels = np.concatenate([ohe_s, ohe_t], axis=0)   

    mask_npy = np.matmul(whole_labels, np.transpose(whole_labels))
    mask = torch.Tensor(mask_npy).to(device)
    mask = torch.triu(mask, diagonal=1) # positive pairs (same class)
    neg_mask = torch.triu(1. - mask, diagonal=1) # negative pairs (different classes)
    
    norm_whole_emb = F.normalize(whole_emb, dim=1)
    cos_sim = torch.matmul(norm_whole_emb, torch.transpose(norm_whole_emb,0,1) )
    dist_matrix = 1. - cos_sim

    # Macro average (Full dataset)
    pos_loss = torch.sum(dist_matrix * mask) / torch.sum(mask) 
    hinge_loss = torch.maximum( neg_mask * (margin - dist_matrix) , torch.zeros_like(neg_mask) )
    neg_loss = torch.sum( hinge_loss ) / torch.sum(neg_mask)

    # Micro average (Per class)
    # pos_loss, neg_loss = 0, 0
    # n_classes = ohe_s.shape[-1]
    # for cl in range(n_classes):
    #     class_mask = whole_labels[:, cl]
    #     pos_loss += torch.sum(dist_matrix[class_mask,:] * mask[class_mask,:]) / torch.sum(mask[class_mask,:]) / n_classes
    #     hinge_loss = torch.maximum( neg_mask[class_mask,:] * (margin - dist_matrix[class_mask,:]) , torch.zeros_like(neg_mask[class_mask,:]) )
    #     neg_loss += torch.sum( hinge_loss ) / torch.sum(neg_mask[class_mask,:]) / n_classes

    return (pos_loss + neg_loss) / 2


def batchContra(emb_source, emb_target, ohe_s, ohe_t, epoch, device ):
    temperature = .5

    # Only inter-domain. Target vs Source
    emb1, emb2 = F.normalize(emb_target), F.normalize(emb_source)
    label1, label2 = ohe_t, ohe_s
    same_samples =  torch.zeros(emb1.shape[0],emb2.shape[0]).to(device) # Ignore comparing sample with itself
    # Target vs (Source + Target)
    # emb1, emb2 = F.normalize(emb_target), F.normalize(torch.cat([emb_source, emb_target],axis=0))
    # label1, label2 = ohe_t, np.concatenate([ohe_s, ohe_t], axis=0)
    # same_samples =  torch.cat((torch.zeros(emb_target.shape[0],emb_source.shape[0]), torch.eye(emb_target.shape[0])),dim=1).to(device)
    # Inter- and intra-domain. (Source + Target)  vs (Source + Target) 
    # emb1 = emb2 = F.normalize(torch.cat([emb_source, emb_target],axis=0))
    # label1 = label2 = np.concatenate([ohe_s, ohe_t], axis=0)
    # same_samples =  torch.eye(emb1.shape[0]).to(device)
    
    mask_npy = np.matmul(label1, np.transpose(label2))
    mask = torch.Tensor(mask_npy).to(device)
    pos_mask = mask - same_samples # positive pairs (same class)
    neg_mask = 1. - mask # negative pairs (different classes)
    nb_pos = pos_mask.sum(dim=1)
    
    cos_sim = torch.matmul(emb1, torch.transpose(emb2,0,1) )
    exp_matrix = torch.exp(cos_sim / temperature)

    # Macro average on classes
    denominator = torch.sum(exp_matrix * neg_mask, dim=1)
    sum_pos = torch.sum(torch.log(exp_matrix / denominator[:, None]) * pos_mask, dim=1)
    loss = -torch.mean( sum_pos / (nb_pos + 1e-8) ) # ignore cases where nb_pos=0

    # Micro average on classes
    # numerator, denominator = 0, 0
    # n_classes = ohe_s.shape[-1]
    # for cl in range(n_classes):
    #     class_mask = whole_labels[:, cl]
    #     denominator = torch.sum(exp_matrix[class_mask] * neg_mask[class_mask], dim=1)
    #     sum_pos = torch.sum(torch.log(exp_matrix[class_mask] / denominator[:, None]) * pos_mask[class_mask], dim=1)
    #     loss -= torch.mean( sum_pos / (nb_pos[class_mask] + 1e-8) ) / n_classes # ignore cases where nb_pos=0 

    return loss

def train_step(model, opt, train_dataloader, test_dataloader, epoch, n_classes, batch_size, criterion, device, num_epochs, valid_dataloader, model_file_name, valid_f1):
    start = time.time()
    model.train()
    train_loss = []
    train_loss_target = []
    train_loss_source = []
    train_contra_loss = []
    train_loss_da = []
    alpha = float(epoch) / num_epochs

    iters = len(train_dataloader)
    for i, sample in  enumerate(train_dataloader):#
        #GET SOURCE AND TARGET TRAINING DATA AND TRAINING LABELS 
        x_train_s, y_train_s, x_train_t, y_train_t = sample
        opt.zero_grad()
        x_train_s = x_train_s.to(device)
        y_train_s = y_train_s.to(device)
        x_train_t = x_train_t.to(device)
        y_train_t = y_train_t.to(device)

        pred_s, discr_s, _, emb_s = model(x_train_s)
        pred_t, discr_t, _, emb_t = model(x_train_t)
        
        ohe_s = F.one_hot(y_train_s, num_classes=n_classes).cpu().detach().numpy()
        ohe_t = F.one_hot(y_train_t, num_classes=n_classes).cpu().detach().numpy()

        # contra_loss_st_ts = simpleDist( emb_s, emb_t, ohe_s, ohe_t, epoch, device )
        contra_loss_st_ts = batchContra( emb_s, emb_t, ohe_s, ohe_t, epoch, device )

        loss_pred_target = criterion( pred_t, y_train_t)
        loss_pred_source = criterion( pred_s, y_train_s)

        gt_da = torch.cat([torch.zeros(discr_s.shape[0]), torch.ones(discr_t.shape[0])], dim=0).long().to(device)
        
        loss_da = torch.mean(  criterion( torch.cat([discr_s,discr_t],dim=0), gt_da)   )
        
        loss_combined = loss_pred_source + loss_pred_target + contra_loss_st_ts + loss_da
        loss_combined.backward()
        opt.step()

        train_loss.append(loss_combined.cpu().detach().numpy())
        train_loss_target.append(loss_pred_target.cpu().detach().numpy())
        train_loss_source.append( loss_pred_source.cpu().detach().numpy() )
        train_contra_loss.append( contra_loss_st_ts.cpu().detach().numpy() )
        train_loss_da.append( loss_da.cpu().detach().numpy())

    end = time.time()
    pred_valid, labels_valid = evaluation(model, valid_dataloader, device)
    f1_val = f1_score(labels_valid, pred_valid, average="weighted")
    f1_2return = valid_f1
    if f1_val > f1_2return:
        torch.save(model.state_dict(), model_file_name)
        f1_2return = f1_val
        pred_test, labels_test = evaluation(model, test_dataloader, device)
        f1 = f1_score(labels_test, pred_test, average="weighted")
        print("Epoch %d | TOT LOSS %.3f | TARGET LOSS %.3f | SOURCE LOSS %.3f | CONTRA LOSS %.3f |DA LOSS %.3f | F1 TARGET %.3f with training time %d"%(epoch, np.mean(train_loss), np.mean(train_loss_target), np.mean(train_loss_source), np.mean(train_contra_loss), np.mean(train_loss_da),f1*100, (end-start) ) )
    else:
        print("TRAIN LOSS at Epoch %d: %.3f | TARGET LOSS %.3f | SOURCE LOSS %.3f | CONTRA LOSS %.3f |DA LOSS %.3f with training time %d"%(epoch, np.mean(train_loss), np.mean(train_loss_target), np.mean(train_loss_source), np.mean(train_contra_loss), np.mean(train_loss_da), (end-start)))
    sys.stdout.flush()
    return f1_2return

def buildPairedSourceTargetData( X_source, Y_source, X_target, Y_target):
    new_X_source = []
    new_X_target = []
    new_Y_source = []
    new_Y_target = []

    for cl in np.unique(Y_source):
        idx_source = np.where(Y_source == cl)[0]
        idx_target = np.where(Y_target == cl)[0]
        n_elem_source = len(idx_source)
        n_elem_target = len(idx_target)

        # Sample with replacement on target domain
        # n_elem = n_elem_source        
        # new_idx_target = np.random.randint(low=0,high=n_elem_target,size=n_elem_source)
        # new_idx_source = np.arange(n_elem_source)
        
        # Upsample smallest domain
        n_elem = max(n_elem_source, n_elem_target)
        if n_elem_source < n_elem_target: #upsample source
            n_repet = n_elem_target // n_elem_source # replicate smaller dataset n_repet times
            remainder = n_elem_target % n_elem_source # then sample remaining without replacement
            new_idx_source = np.concatenate( ( np.tile(shuffle(np.arange(n_elem_source)),n_repet), \
                                                np.random.choice(n_elem_source, size=remainder, replace=False)), axis=0)
            new_idx_target =  np.arange(n_elem_target)
        else: #upsample target (usually the case)
            n_repet = n_elem_source // n_elem_target 
            remainder = n_elem_source % n_elem_target
            new_idx_target = np.concatenate( ( np.tile(shuffle(np.arange(n_elem_target)),n_repet), \
                                                np.random.choice(n_elem_target, size=remainder, replace=False)), axis=0)
            new_idx_source =  np.arange(n_elem_source)

        t_source = X_source[idx_source][new_idx_source]
        new_X_source.append( t_source )

        t_target = X_target[idx_target][new_idx_target]
        new_X_target.append( t_target )

        new_Y_source.append( np.ones(n_elem) * cl  )
        new_Y_target.append( np.ones(n_elem) * cl  )
    
    new_X_source = np.concatenate(new_X_source,axis=0)
    new_X_target = np.concatenate(new_X_target,axis=0)
    new_Y_source = np.concatenate(new_Y_source,axis=0)
    new_Y_target = np.concatenate(new_Y_target,axis=0)

    new_X_source, new_X_target, new_Y_source, new_Y_target = shuffle(new_X_source, new_X_target, new_Y_source, new_Y_target)
    return new_X_source, new_Y_source, new_X_target, new_Y_target


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        loss = (self.alpha[targets] * (1 - pt) ** self.gamma * ce_loss).mean()
        return loss


def classWeights(y_train):
    class_counts = np.bincount(y_train.astype(int))
    total_samples = len(y_train)

    class_weights = []
    for count in class_counts:
        weight = 1 / (count / total_samples)
        class_weights.append(weight)

    return torch.tensor(class_weights, dtype=torch.int64)


# ################################
# Script main body

# Testing of script arguments number

source_year = int( sys.argv[1] )
id_ = int( sys.argv[2] )
target_year = int( sys.argv[3] )
rng_seed = int(sys.argv[4]) if len(sys.argv) > 4 else 42

print(f'(Random seed set to {rng_seed})')
torch.manual_seed(rng_seed)
np.random.seed(rng_seed)

path_source = f'./DATA/' #./DATA_CVL_{source_year}/
path_target = f'./DATA/'

X_train_target = np.load("%strain_data_%d_%d.npy"%(path_target, id_, target_year))
Y_train_target = np.load("%strain_label_%d_%d.npy"%(path_target, id_, target_year))

X_train_source = np.load("%sdata_%d.npy"%(path_source, source_year))
Y_train_source = np.load("%sgt_data_%d.npy"%(path_source, source_year))
if Y_train_source.ndim > 1 and Y_train_source.shape[1] > 2:
    Y_train_source = Y_train_source[:,2]

valid_data = np.load("%svalid_data_%d_%d.npy"%(path_target, id_,target_year)) 
valid_label = np.load("%svalid_label_%d_%d.npy"%(path_target, id_,target_year))

X_test_target = np.load("%stest_data_%d_%d.npy"%(path_target, id_, target_year))
Y_test_target = np.load("%stest_label_%d_%d.npy"%(path_target, id_, target_year))

Y_train_target = Y_train_target - 1
Y_train_source = Y_train_source - 1
Y_test_target = Y_test_target - 1
valid_label = valid_label - 1


n_elems = X_train_source.shape[0]
n_classes = len(np.unique(Y_train_source))
seq_len = X_train_source.shape[2]

# X_train_target, Y_train_target = rebalance_data(n_elems, X_train_target, Y_train_target) # Handled on buildPairedSourceTargetData()

proj_dim = 128
batch_size = 512
learning_rate = 0.0001
num_epochs = 300

test_dataloader = createDataLoader(X_test_target, Y_test_target, shuffling=False, batch_size = 2048)


x_valid = torch.tensor(valid_data, dtype=torch.float32)
y_valid = torch.tensor(valid_label, dtype=torch.int64)

valid_dataset = TensorDataset(x_valid, y_valid)
valid_dataloader = DataLoader(valid_dataset, shuffle=False, batch_size=2048)

beta = 1.0


device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = TempCNNWP([X_train_source.shape[1], X_train_source.shape[2]], proj_dim=proj_dim, num_classes=n_classes)
#embed_dim = 128
#model = TransformerEncoder(n_classes, seq_len, embed_dim)

model = model.to(device)

# Classification loss
# 1) CrossEntropy
criterion = nn.CrossEntropyLoss()
# 2) Focal Loss
# class_weights = classWeights(np.concatenate([Y_train_source, Y_train_target], axis=0)).to(device)
# criterion = FocalLoss(alpha=class_weights, gamma=2)

opt = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)

model_file_name = "model_sda_%d_%d_%d.pth"%(source_year, id_, target_year)
valid_f1 = 0.0
for epoch in range(num_epochs):
    X_train_source, Y_train_source = shuffle(X_train_source, Y_train_source)
    X_train_target, Y_train_target = shuffle(X_train_target, Y_train_target)
    X_train_source_new, Y_train_source_new, X_train_target_new, Y_train_target_new = buildPairedSourceTargetData( X_train_source, Y_train_source, X_train_target, Y_train_target )
    train_dataloader = createDataLoaderDouble(X_train_source_new, X_train_target_new, Y_train_source_new, Y_train_target_new, batch_size = batch_size)
    valid_f1 = train_step(model, opt, train_dataloader, test_dataloader, epoch, n_classes, batch_size, criterion, device, num_epochs, valid_dataloader, model_file_name, valid_f1)
