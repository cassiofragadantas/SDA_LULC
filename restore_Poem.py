import torch
import torch.nn as nn
import sys
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import sys
from sklearn.utils import shuffle
#from model_transformer import TransformerEncoder
#from model_pytorch import TempCNNDisentangleV2
#from model_pytorch import TempCNNDisentangleV3
from model_pytorch import TempCNNPoem
import time
from sklearn.metrics import f1_score, accuracy_score
import os

def evaluation(model, dataloader, device):
    model.eval()
    tot_pred = []
    tot_labels = []
    embs = []
    for x_batch, y_batch in dataloader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        pred, _, _, emb ,_ = model(x_batch)
        pred_npy = np.argmax(pred.cpu().detach().numpy(), axis=1)
        tot_pred.append( pred_npy )
        tot_labels.append( y_batch.cpu().detach().numpy())
        embs.append( emb.cpu().detach().numpy() )
    tot_pred = np.concatenate(tot_pred)
    tot_labels = np.concatenate(tot_labels)
    embs = np.concatenate(embs,axis=0)
    return tot_pred, tot_labels, embs

source_year = int(sys.argv[1])
target_year = int(sys.argv[2])
dataset = sys.argv[3]

device = 'cuda' if torch.cuda.is_available() else 'cpu'

prefix_path = "data_%s/"%dataset

tot_avg_f1 = []
tot_micro_f1 = []
tot_macro_f1 = []
tot_acc = []
tot_f1_class = None

n_splits = 5

for i in range(n_splits):

    #model_combined_source_target_dis_V2_koumbia_2020_0_2021.pth
    file_name = "poem_%s_%s_%d_%s.pth"%(dataset, source_year, i, target_year)
    print(file_name)
    #file_name = "model_combined_source_target_dis_V4_%s_%s_%d_%s.pth"%(dataset, source_year, i, target_year)
    #file_name = "model_combined_source_target_dis_V4_flatM_%s_%s_%d_%s.pth"%(dataset, source_year, i, target_year)
    
    test_data = np.load(prefix_path+"test_data_%d_%d.npy"%(i,target_year)) 
    test_label = np.load(prefix_path+"test_label_%d_%d.npy"%(i,target_year))-1

    if not os.path.exists(file_name):
        continue

    x_test = torch.tensor(test_data, dtype=torch.float32)
    y_test = torch.tensor(test_label, dtype=torch.int64)
    test_dataset = TensorDataset(x_test, y_test)
    test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=2048)

    n_classes = len(test_label)
    #model = TempCNNDisentangleV2(test_data.shape[2], test_data.shape[1], n_classes).to(device)
    model = TempCNNPoem(n_classes).to(device)
    print(file_name)
    model.load_state_dict(torch.load(file_name))
    print("model loaded")
    pred, labels, embs = evaluation(model, test_dataloader, device)
    temp_f1_avg = f1_score(labels, pred, average="weighted")

    tot_avg_f1.append( temp_f1_avg )
    tot_micro_f1.append( f1_score(labels, pred, average="micro") )
    tot_macro_f1.append( f1_score(labels, pred, average="macro") )
    tot_acc.append( accuracy_score(labels, pred))

    if tot_f1_class is None:
        tot_f1_class = f1_score(labels, pred, average=None)
    else:
        tot_f1_class = tot_f1_class + f1_score(labels, pred, average=None)
    
    np.save("poem_%d_embeddings.npy"%i, embs)

tot_f1_class = tot_f1_class / n_splits

    
#print(tot_avg_f1)
print("average F1 %.2f $\pm$ %.2f"%(np.mean(tot_avg_f1)*100, np.std(tot_avg_f1)*100 ))
print("micro F1 %.2f $\pm$ %.2f"%(np.mean(tot_micro_f1)*100, np.std(tot_micro_f1)*100 ))
print("macro F1 %.2f $\pm$ %.2f"%(np.mean(tot_macro_f1)*100, np.std(tot_macro_f1)*100 ))
print("acc %.2f $\pm$ %.2f"%(np.mean(tot_acc)*100, np.std(tot_acc)*100 ))
print( ' '.join( ["%.2f"%(el*100) for el in tot_f1_class]) )

