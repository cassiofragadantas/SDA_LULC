import sys
import numpy as np
import sys
#from model_transformer import TransformerEncoder
import time
from sklearn.metrics import f1_score, accuracy_score
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
dataset = sys.argv[4]

model_path = './DATA/'
data_path = f'./DATA_{dataset}/'

tot_avg_f1 = []
tot_micro_f1 = []
tot_acc = []
tot_perclass_f1 = []
for i in range(5):
    pred = None
    #DIRECT TRANSFER
    if model_type == 0:
        file_name = "%srf_model_direct_prediction_%s_%s_%s.npy"%(model_path, source_year,target_year,i)
    #FULL TARGET
    if model_type == 1:
        file_name = "%srf_model_full_target_prediction_%s_%s.npy"%(model_path, target_year,i)
    #COMBINED SOURCE+TARGET
    if model_type == 2:
        file_name = "%srf_model_combined_prediction_%s_%s_%s.npy"%(model_path, source_year, target_year, i)
    
    test_data = np.load("%stest_data_%d_%d.npy"%(data_path,i,target_year))
    nrow, nts, nc = test_data.shape
    test_data = np.reshape(test_data, (nrow, nts * nc))
    test_label = np.load("%stest_label_%d_%d.npy"%(data_path,i,target_year))
    print(file_name)
    if not os.path.exists(file_name):
        continue
    
    pred = np.load(file_name)
    tot_avg_f1.append( f1_score(test_label, pred, average="weighted") )
    tot_micro_f1.append( f1_score(test_label, pred, average="micro") )
    tot_acc.append( accuracy_score(test_label, pred) )
    tot_perclass_f1.append( f1_score(test_label, pred, average=None) )

print("average F1 %.2f $\pm$ %.2f"%(np.mean(tot_avg_f1)*100, np.std(tot_avg_f1)*100 ))
print("micro F1 %.2f $\pm$ %.2f"%(np.mean(tot_micro_f1)*100, np.std(tot_micro_f1)*100 ))
print("acc %.2f $\pm$ %.2f"%(np.mean(tot_acc)*100, np.std(tot_acc)*100 ))
print("per-class F1 %s $\pm$ %s"
      %(np.array2string(np.mean(np.stack(tot_perclass_f1),axis=0)*100, precision=2), 
        np.array2string(np.std(np.stack(tot_perclass_f1),axis=0)*100, precision=2 )))

