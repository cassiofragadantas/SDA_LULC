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
from model_pytorch import TempCNN, TempCNNWP2

#from model_transformer import TransformerEncoder
from torch.optim.lr_scheduler import CosineAnnealingLR, CyclicLR, ExponentialLR, StepLR

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




def computeCLoss(emb1, emb2, lab1, lab2, device, temperature = 1.0):
    eps=torch.tensor(1e-10).to(device)

    # BUILD THE 1to1 MASK THAT ASSOCIATE EMB1 TO EMB2 IF THEY HAVE THE SAME CLASS
    mask_npy = np.matmul(lab1, np.transpose(lab2))
    mask = torch.tensor(mask_npy).to(device)

    '''
    #EUCL DISTANCE BASED SIMILARITY
    eucl_d = torch.cdist(emb1, emb2) / emb1.shape[1]
    #exp_sim = torch.exp(-eucl_d / temperature)
    #exp_sim = torch.exp(-eucl_d )
    vals, _ = torch.max(eucl_d * mask, dim=1)
    return torch.mean( vals )

    '''
    #COSINE BASED SIMILARITY
    norm_emb1 = F.normalize(emb1, dim=1)
    norm_emb2 = F.normalize(emb2, dim=1)
    cos_sim = torch.matmul(norm_emb1, torch.transpose(norm_emb2,0,1) )
    #exp_sim = torch.exp(cos_sim / temperature)
    exp_sim = torch.exp(torch.div( cos_sim, temperature) )
    

    #NEGATIVE MASK THAT IS THE COMPLEMENT OF THE FIRST MASK
    neg_mask = torch.tensor(1. - mask_npy).to(device)
    #neg_mask_max = neg_mask * sys.float_info.max

    #  WE CONSIDER AS NUMERATOR FOR THE CONTRASTIVE LEARNING APPROACH THE SUM OF ALL THE POSITIVE MATCHES FROM MASK BINARY MASK    
    num = torch.sum(exp_sim * mask, dim=1) + eps
    #num = torch.div( num, torch.sum(mask, dim=1) )
    

    #COMPUTE MIN SIMILARITY
    #min_sim, _ = torch.min(  exp_sim + neg_mask, dim=1 )
    #num = min_sim

    #Take the per row sum of all the negative pairs
    #den = torch.sum(exp_sim * neg_mask, dim=1) #+ eps
    den = torch.sum(exp_sim, dim=1) #+ eps
    #print("num ", num)
    #print("den ", den)
    detailed_loss = -torch.log( torch.div( num, den ) )
    loss = torch.mean( detailed_loss )
    #print("loss ",detailed_loss)
    #print("========")
    #CONTRASTIVE LOSS
    return loss
    
    

def contraSiameseLossV2(emb_source, emb_target, ohe_s, ohe_t, epoch, device):
    t_min = 0.02
    t_max = 1.0
    T = 50
    #temperature = (t_max - t_min) * (1 + np.cos(2 * epoch * math.pi/ T ) ) / 2 + t_min
    temperature = 1.
    whole_emb = torch.cat([emb_source, emb_target],axis=0)
    whole_labels = np.concatenate([ohe_s, ohe_t], axis=0)

    loss = computeCLoss(whole_emb, whole_emb, whole_labels, whole_labels, device, temperature=temperature)
    
    
    loss_st = computeCLoss(emb_source, emb_target, ohe_s, ohe_t, device, temperature=temperature)
    loss_ts = computeCLoss(emb_target, emb_source, ohe_t, ohe_s, device, temperature=temperature)
    loss_tt = computeCLoss(emb_target, emb_target, ohe_t, ohe_t, device, temperature=temperature)
    loss_ss = computeCLoss(emb_source, emb_source, ohe_s, ohe_s, device, temperature=temperature)
    '''
    '''
    #exit()
    #return (loss_st + loss_ts + loss_tt)/3#loss#(loss_st + loss_ts + loss_tt) / 3
    #return (loss_st + loss_ts)/2, loss_tt
    #return (loss_st + loss_ts)/2, loss_tt
    #print(loss_st)
    #print(loss_ts)
    #print("=======")
    #return (loss_st + loss_ts)/2, loss_tt
    return (loss_st+ loss_ts)/2, loss_tt
    #return (loss_ts + loss_st)/2
    #return loss_st, loss_tt


def randomChange(vals):
    vec = np.random.rand(vals.shape[0])
    vec = vec*2 - 1
    result = (( vec + vals) > .5).astype("int")
    return result


def getIdxTarget(values, y_train_t ):
    temp_idx = []
    for v in values:
        temp_idx.append( np.where(y_train_t == v)[0] )
    return np.concatenate(temp_idx,axis=0)


def contraSiameseLossV1(y_train_s, y_train_t, ohe_s, ohe_t, emb_s, emb_t, epoch, device):
    t_min = 0.1
    t_max = .5
    T = 50
    margin = (t_max - t_min) * (1 + np.cos(2 * epoch * math.pi/ T ) ) / 2 + t_min
    norm_emb1 = F.normalize(emb_s, dim=1)
    norm_emb2 = F.normalize(emb_t, dim=1)
    '''
    indicator = (y_train_s.cpu().detach().numpy() == y_train_t.cpu().detach().numpy() ).astype("int")
    indicator = torch.tensor(indicator).to(device)
    cos_sim = torch.sum(norm_emb1 * norm_emb2,dim=1)
    loss = indicator * ( 1. - cos_sim ) + (1. - indicator) * cos_sim#torch.max(  cos_sim - margin, torch.zeros_like(cos_sim).to(device))
    return torch.mean(loss)
    '''

    cos_sim = torch.matmul(norm_emb1, torch.transpose(norm_emb2,0,1) )
    indicator_matrix = torch.matmul(ohe_s.float(), torch.transpose(ohe_t,0,1).float() )

    cos_sim_reshape = cos_sim.view(-1)
    indicator_matrix_reshape = indicator_matrix.view(-1)
    #print(cos_sim.shape)
    #print(cos_sim_reshape.shape)
    #exit()
    loss = indicator_matrix_reshape * ( 1. - cos_sim_reshape ) + (1. - indicator_matrix_reshape) * torch.max(  cos_sim_reshape - margin, torch.zeros_like(indicator_matrix_reshape).to(device))
    
    #print( torch.count_nonzero(loss) / loss.shape[0])
    
    #return torch.sum(loss) / torch.count_nonzero(loss) 
    return torch.mean(loss)

def pretrain_step(model, train_dataloader, opt, epoch, device):
    model.train()
    cumul_loss = []
    for sample in  train_dataloader:
        opt.zero_grad()
        orig_x, corrupted_x, mask = sample
        orig_x = orig_x.to(device)
        corrupted_x = corrupted_x.to(device)
        mask = mask.to(device)

        _,_,reco,_ = model(corrupted_x)
        diff_squared = torch.square( (reco - orig_x) * mask)
        #diff_squared = torch.abs( (reco - orig_x) * mask)
        sum_ = torch.sum( diff_squared )
        loss = sum_ /  ( torch.count_nonzero(mask) / orig_x.shape[1] )
        loss.backward()
        opt.step()
        cumul_loss.append( loss.cpu().detach().numpy())
    print("epoch %d MASKED RMSE %.3f"%(epoch, np.mean(cumul_loss)) )

def train_step(model, opt, scheduler, train_dataloader, test_dataloader, epoch, n_classes, batch_size, criterion, criterion2, device, num_epochs):
    model.train()
    train_loss = []
    train_loss_target = []
    train_loss_source = []
    train_contra_loss = []
    train_loss_da = []
    alpha = float(epoch) / num_epochs
    margin = 0.2


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
        
        
        target_p = pred_t.cpu().detach().numpy()
        target_p = np.argmax(target_p,axis=1)
        #SELECT EXAMPLES THAT ARE CORRECTLY CLASSIFIED BY THE MODEL ON THE TRAINING TARGET DATA
        check = (y_train_t.cpu().detach().numpy() == target_p).astype("int")

        #SELECT EXAMPLES THAT ARE CORRECTLY CLASSIFIED BY THE MODEL ON THE TRAINING SOURCE DATA
        source_p = pred_s.cpu().detach().numpy()
        source_p = np.argmax(source_p,axis=1)
        check_s = (y_train_s.cpu().detach().numpy() == source_p).astype("int")

        '''
        idx_s = np.where(check_s == 1)[0]
        emb_s = emb_s[idx_s]
        y_train_s_subset = y_train_s[idx_s]
        ohe_s = F.one_hot(y_train_s_subset, num_classes=n_classes)
        
        idx_t = getIdxTarget(np.unique( y_train_s_subset.cpu().detach().numpy()),  y_train_t.cpu().detach().numpy() )
        emb_t = emb_t[idx_t]
        y_train_t_subset = y_train_t[idx_t]
        ohe_t = F.one_hot(y_train_t_subset, num_classes=n_classes)
        '''
        ohe_s = F.one_hot(y_train_s, num_classes=n_classes).cpu().detach().numpy()
        ohe_t = F.one_hot(y_train_t, num_classes=n_classes).cpu().detach().numpy()
        
        contra_loss_st_ts, contra_loss_tt = contraSiameseLossV2(emb_s, emb_t, ohe_s, ohe_t, epoch, device)                
        
        loss_pred_target = criterion( pred_t, y_train_t)
        loss_pred_source = criterion( pred_s, y_train_s)

        #contra_loss1 = contraSiameseLossV1(y_train_s, y_train_t, ohe_s, ohe_t, emb_s, emb_t, epoch, device)

        '''
        loss_pred_target = torch.mean(criterion( pred_t, y_train_t) * torch.tensor(check).to(device)) #torch.mean( criterion(y_train_s, pred_s) )
        loss_pred_target_full = torch.mean(criterion( pred_t, y_train_t))


        loss_pred_source = torch.mean( criterion( pred_s, y_train_s) * torch.tensor(check_s).to(device) )
        loss_pred_source_full = torch.mean(criterion( pred_s, y_train_s))
        '''
        
        mask_gt_da = np.concatenate([ohe_s, ohe_t],axis=0)
        mask_gt_da = torch.tensor(mask_gt_da).to(device)

        gt_da = torch.cat([torch.zeros(discr_s.shape[0], discr_s.shape[1]), torch.ones(discr_t.shape[0], discr_t.shape[1])], dim=0).float().to(device)


        bce_loss = criterion2( torch.cat([discr_s,discr_t],dim=0), gt_da)
        loss_da = torch.mean( torch.sum(bce_loss * mask_gt_da, axis=1) )


        #gt_da = torch.cat([torch.zeros(discr_s.shape[0]), torch.ones(discr_t.shape[0])], dim=0).long().to(device)
        
        #loss_da = torch.mean(  criterion( torch.cat([discr_s,discr_t],dim=0), gt_da)   )
        

        #loss_combined = loss_pred_target + loss_pred_source + contra_loss_st_ts
        #loss_combined= (1-alpha) * ( loss_pred_source + contra_loss_st_ts + loss_da) + alpha * loss_pred_target
        loss_combined = (1-alpha) * (loss_pred_source + loss_da + contra_loss_st_ts) + alpha * (loss_pred_target) + contra_loss_tt #(1-alpha) * (loss_pred_source + contra_loss_st_ts + loss_da) + alpha * (loss_pred_target)  # + 
        #loss_combined= (1-alpha)*(loss_pred_source + contra_loss_st_ts) + alpha * loss_pred_target   #contra_loss_st_ts #+  
        #loss_combined = loss_pred_target_full + loss_pred_source + contra_loss1
        loss_combined.backward()
        opt.step()
        #train_loss.append(loss_pred_source.cpu().detach().numpy())
        train_loss.append(loss_combined.cpu().detach().numpy())
        train_loss_target.append(loss_pred_target.cpu().detach().numpy())
        train_loss_source.append( loss_pred_source.cpu().detach().numpy() )
        train_contra_loss.append( contra_loss_st_ts.cpu().detach().numpy() )
        train_loss_da.append( loss_da.cpu().detach().numpy())

    #scheduler.step()
    model.eval()
    tot_pred = []
    tot_labels = []
    for x_batch, y_batch in test_dataloader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        pred, _, _, _ = model(x_batch)
        pred_npy = np.argmax(pred.cpu().detach().numpy(), axis=1)
        tot_pred.append( pred_npy )
        tot_labels.append( y_batch.cpu().detach().numpy())
    
    tot_pred = np.concatenate(tot_pred)
    tot_labels = np.concatenate(tot_labels)
    f1 = f1_score(tot_labels,tot_pred,average="weighted")    

    print("Epoch %d | TOT LOSS %.3f | TARGET LOSS %.3f | SOURCE LOSS %.3f | CONTRA LOSS %.3f |DA LOSS %.3f | F1 TARGET %.3f"%(epoch, np.mean(train_loss), np.mean(train_loss_target), np.mean(train_loss_source), np.mean(train_contra_loss), np.mean(train_loss_da),f1*100 ) )
    #print("LR ",opt.param_groups[0]["lr"])
    sys.stdout.flush()


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
        t_source = X_source[idx_source]

        new_X_source.append( t_source )
        new_idx_target = np.random.randint(low=0,high=n_elem_target,size=n_elem_source)
        
        t_target = X_target[idx_target][new_idx_target]
        new_X_target.append( t_target )

        new_Y_source.append( np.ones(n_elem_source) * cl  )
        new_Y_target.append( np.ones(n_elem_source) * cl  )
    
    new_X_source = np.concatenate(new_X_source,axis=0)
    new_X_target = np.concatenate(new_X_target,axis=0)
    new_Y_source = np.concatenate(new_Y_source,axis=0)
    new_Y_target = np.concatenate(new_Y_target,axis=0)

    new_X_source, new_X_target, new_Y_source, new_Y_target = shuffle(new_X_source, new_X_target, new_Y_source, new_Y_target)
    return new_X_source, new_Y_source, new_X_target, new_Y_target




def generatedCorruptedElement(elem):
    nchannels, nts = elem.shape
    corrupted = []
    mask = []
    for i in range(nts):
        if np.random.rand(1) > .5:
            # MODIFY
            '''
            if np.random.rand(1) > .5:
            #ADD POSITIVE NOISE
                corrupted.append( elem[:,i] + np.random.uniform(low=.5, high=.5,size=(nchannels)) )
            else:
            #REMOVE POSITIVE NOISE
                corrupted.append( elem[:,i] - np.random.uniform(low=0, high=.5,size=(nchannels)) )
            '''
            corrupted.append( elem[:,i] + np.random.uniform(low=-0.5, high=0.5,size=(nchannels)) )
            mask.append(np.ones(nchannels))
        else:
            #DO NOT MODIFY
            corrupted.append( elem[:,i] )
            mask.append(np.zeros(nchannels))
    corrupted = np.stack(corrupted, axis=1)
    mask = np.stack(mask, axis=1)
    return corrupted, mask

def generateCorruptedData(X_train, batch_size):
    nrow, nchannels, nts = X_train.shape
    X_train = shuffle(X_train)
    tmask = np.random.rand(nrow, nts)     
    tmask = (tmask > .5).astype("int")
    mask = [tmask for _ in range(nchannels)]
    mask = np.array(mask)
    data_mask = np.moveaxis(mask, (0,1,2), (1,0,2))
    noise = np.random.uniform(low=-0.5, high=0.5,size=(nrow, nchannels, nts))
    #noise_pos = np.random.uniform(low=0, high=0.5,size=(nrow//2, nchannels, nts))
    #noise_neg = np.random.uniform(low=-0.5, high=0,size=(nrow-(nrow//2), nchannels, nts))
    #noise = np.concatenate([noise_pos, noise_neg],axis=0)
    data_corrupted = X_train + noise * data_mask
    #data_corrupted = X_train * ( 1 - data_mask)
    
    '''
    data_corrupted = []
    data_mask = []
    for elem in X_train:
        corrupted, mask = generatedCorruptedElement(elem)
        data_corrupted.append( corrupted )
        data_mask.append( mask )
    '''


    #data_corrupted = np.array(data_corrupted)
    #data_mask = np.array(data_mask)

    X_train, data_corrupted, data_mask = shuffle(X_train, data_corrupted, data_mask)
    
    x_train = torch.tensor(X_train, dtype=torch.float32)
    data_corrupted = torch.tensor(data_corrupted, dtype=torch.float32)    
    data_mask = torch.tensor(data_mask, dtype=torch.float32)

    train_dataset = TensorDataset(x_train, data_corrupted, data_mask)
    return DataLoader(train_dataset, shuffle=True, batch_size=batch_size)


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

X_train_target = np.load("./DATA/train_data_%d_%d.npy"%(id_, target_year))
Y_train_target = np.load("./DATA/train_label_%d_%d.npy"%(id_, target_year))


X_train_source = np.load("./DATA/data_%d.npy"%(source_year))
Y_train_source = np.load("./DATA/gt_data_%d.npy"%source_year)
if Y_train_source.ndim > 1 and Y_train_source.shape[1] > 2:
    Y_train_source = Y_train_source[:,2]

X_test_target = np.load("./DATA/test_data_%d_%d.npy"%(id_, target_year))
Y_test_target = np.load("./DATA/test_label_%d_%d.npy"%(id_, target_year))

Y_train_target = Y_train_target - 1
Y_train_source = Y_train_source - 1
Y_test_target = Y_test_target - 1


n_elems = X_train_source.shape[0]
n_classes = len(np.unique(Y_train_source))
seq_len = X_train_source.shape[2]

X_train_target, Y_train_target = rebalance_data(n_elems, X_train_target, Y_train_target)

batch_size = 512
learning_rate = 0.0001
num_epochs = 250#250

train_dataloader = createDataLoaderDouble(X_train_source, X_train_target, Y_train_source, Y_train_target, batch_size = batch_size)
test_dataloader = createDataLoader(X_test_target, Y_test_target, shuffling=False, batch_size = 2048)




beta = 1.0

#train_batch_size = 256
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = TempCNNWP2([X_train_source.shape[1], X_train_source.shape[2]], num_classes=n_classes)
#embed_dim = 128
#model = TransformerEncoder(n_classes, seq_len, embed_dim)

model = model.to(device)

#criterion = nn.CrossEntropyLoss(reduction='none')
criterion = nn.CrossEntropyLoss()
criterion2 = torch.nn.BCEWithLogitsLoss(reduction='none')

opt = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
#opt = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)

#opt2 = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

#opt = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=5e-4)

#scheduler = ExponentialLR(opt, gamma=0.99)#
#scheduler = CosineAnnealingLR(opt, num_epochs,eta_min=0.00001)
scheduler = StepLR(opt, step_size=50, gamma=0.1)
#scheduler = CyclicLR(opt, base_lr=0.001, max_lr=0.1,step_size_up=5,mode="exp_range",gamma=0.85,cycle_momentum=False)


'''
#X_train_target_copy = np.array(X_train_target)
X_train_target_copy = np.concatenate([X_train_target, X_train_source],axis=0)
for epoch in range(50):
    train_loader_corrupted = generateCorruptedData(X_train_target_copy, 256)
    pretrain_step(model, train_loader_corrupted, opt2, epoch, device) 
'''

for epoch in range(num_epochs):
    X_train_source, Y_train_source, X_train_target, Y_train_target = shuffle(X_train_source, Y_train_source, X_train_target, Y_train_target)
    X_train_target, Y_train_target = rebalance_data(n_elems, X_train_target, Y_train_target)
    train_dataloader = createDataLoaderDouble(X_train_source, X_train_target, Y_train_source, Y_train_target, batch_size = batch_size)
    
    train_step(model, opt, scheduler, train_dataloader, test_dataloader, epoch, n_classes, batch_size, criterion, criterion2, device, num_epochs)
    