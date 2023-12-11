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
import os

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
target_year = int(sys.argv[2])
model_type = int(sys.argv[3])

device = 'cuda' if torch.cuda.is_available() else 'cpu'


tot_avg_f1 = []
tot_micro_f1 = []
tot_perclass_f1 = []
for i in range(5):
    file_name = None
    if model_type == 0:
        file_name = "model_direct_transfer_%s_%d_%s.pth"%(source_year, i, target_year)
    if model_type == 1:
        file_name = "model_full_target_%d_%s.pth"%(i, target_year)
    if model_type == 2:
        file_name = "model_combined_source_target_%s_%d_%s.pth"%(source_year, i, target_year)
    path_name = './DATA/'
    test_data = np.load("%stest_data_%d_%d.npy"%(path_name,i,target_year))
    test_label = np.load("%stest_label_%d_%d.npy"%(path_name,i,target_year))-1

    if not os.path.exists(file_name):
        continue

    x_test = torch.tensor(test_data, dtype=torch.float32)
    y_test = torch.tensor(test_label, dtype=torch.int64)
    test_dataset = TensorDataset(x_test, y_test)
    test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=2048)

    n_classes = len(test_label)
    model = TempCNN(n_classes).to(device)
    print(file_name)
    model.load_state_dict(torch.load(file_name))
    print("model loaded")
    pred, labels = evaluation(model, test_dataloader, device)
    tot_avg_f1.append( f1_score(labels, pred, average="weighted") )
    tot_micro_f1.append( f1_score(labels, pred, average="micro") )
    tot_perclass_f1.append( f1_score(labels, pred, average=None) )

print("average F1 %.2f $\pm$ %.2f"%(np.mean(tot_avg_f1)*100, np.std(tot_avg_f1)*100 ))
print("micro F1 %.2f $\pm$ %.2f"%(np.mean(tot_micro_f1)*100, np.std(tot_micro_f1)*100 ))
print("per-class F1 %s $\pm$ %s"
      %(np.array2string(np.mean(np.stack(tot_perclass_f1),axis=0)*100, precision=2), 
        np.array2string(np.std(np.stack(tot_perclass_f1),axis=0)*100, precision=2 )))

