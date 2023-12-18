### To implemente the Transformer Framework I used the code from this website : https://www.kaggle.com/code/arunmohan003/transformer-from-scratch-using-pytorch

import torch
import torch.nn as nn
import sys
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import sys
from sklearn.utils import shuffle
#from model_transformer import TransformerEncoder
from model_pytorch import TempCNN, Inception
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


id_ = int(sys.argv[1])
target_year = int(sys.argv[2])
rng_seed = int(sys.argv[3]) if len(sys.argv) > 3 else 42

print(f'(Random seed set to {rng_seed})')
torch.manual_seed(rng_seed)
np.random.seed(rng_seed)

path_target = f'./DATA/' #./DATA_CVL_{target_year}/

train_data = np.load("%strain_data_%d_%d.npy"%(path_target, id_, target_year))
train_label = np.load("%strain_label_%d_%d.npy"%(path_target, id_, target_year))

valid_data = np.load("%svalid_data_%d_%d.npy"%(path_target, id_,target_year)) 
valid_label = np.load("%svalid_label_%d_%d.npy"%(path_target, id_,target_year))

test_data = np.load("%stest_data_%d_%d.npy"%(path_target, id_, target_year))
test_label = np.load("%stest_label_%d_%d.npy"%(path_target, id_, target_year))

print(train_data.shape)
print(train_label.shape)

n_classes = len( np.unique(train_label))

train_label = train_label - 1
test_label = test_label - 1
valid_label = valid_label - 1

x_train = torch.tensor(train_data, dtype=torch.float32)
y_train = torch.tensor(train_label, dtype=torch.int64)

x_valid = torch.tensor(valid_data, dtype=torch.float32)
y_valid = torch.tensor(valid_label, dtype=torch.int64)

x_test = torch.tensor(test_data, dtype=torch.float32)
y_test = torch.tensor(test_label, dtype=torch.int64)

train_dataset = TensorDataset(x_train, y_train)
valid_dataset = TensorDataset(x_valid, y_valid)
test_dataset = TensorDataset(x_test, y_test)

train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=256)
valid_dataloader = DataLoader(valid_dataset, shuffle=False, batch_size=2048)
test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=512)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

seq_length = train_data.shape[2]
embedding_size = 64
#model = Inception(n_classes).to(device)
model = TempCNN(n_classes).to(device)


learning_rate = 0.0001
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(params=model.parameters(), lr=learning_rate)
#optimizer = torch.optim.SGD(params=model.parameters(), lr=learning_rate)
#optimizer = torch.optim.AdamW(params=model.parameters(), lr=learning_rate)

epochs = 300
# Loop through the data
valid_f1 = 0.0
for epoch in range(epochs):
    start = time.time()
    model.train()
    tot_loss = 0.0
    den = 0
    for x_batch, y_batch in train_dataloader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        optimizer.zero_grad()
        pred, _ = model(x_batch)
        loss = loss_fn(pred, y_batch)

        loss.backward() # backward pass: backpropagate the prediction loss
        optimizer.step() # gradient descent: adjust the parameters by the gradients collected in the backward pass
        tot_loss+= loss.cpu().detach().numpy()
        den+=1.
        #break
    
    end = time.time()
    pred_valid, labels_valid = evaluation(model, valid_dataloader, device)
    f1_val = f1_score(labels_valid, pred_valid, average="weighted")
    if f1_val > valid_f1:
        torch.save(model.state_dict(), "model_full_target_%d_%d.pth"%(id_, target_year))
        valid_f1 = f1_val
        pred_test, labels_test = evaluation(model, test_dataloader, device)
        f1 = f1_score(labels_test, pred_test, average="weighted")
        print("TRAIN LOSS at Epoch %d: %.4f with acc on TEST TARGET SET %.2f with training time %d"%(epoch, tot_loss/den, 100*f1,(end-start)))
    else:
        print("TRAIN LOSS at Epoch %d: %.4f"%(epoch, tot_loss/den))
    sys.stdout.flush()
    



