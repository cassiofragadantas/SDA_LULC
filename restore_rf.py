import sys
import numpy as np
import sys
#from model_transformer import TransformerEncoder
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

tot_avg_f1 = []
tot_micro_f1 = []
for i in range(5):
    pred = None
    #DIRECT TRANSFER
    if model_type == 0:
        file_name = "rf_model_direct_prediction_%s_%s_%s.npy"%(source_year,target_year,i)
    #FULL TARGET
    if model_type == 1:
        file_name = "rf_model_full_target_prediction_%s_%s.npy"%(target_year,i)
    #COMBINED SOURCE+TARGET
    if model_type == 2:
        file_name = "rf_model_combined_prediction_%s_%s_%s.npy"%(source_year, target_year, i)
    
    path_name = './DATA/'
    test_data = np.load("%stest_data_%d_%d.npy"%(path_name,i,target_year))
    nrow, nts, nc = test_data.shape
    test_data = np.reshape(test_data, (nrow, nts * nc))
    test_label = np.load("%stest_label_%d_%d.npy"%(path_name,i,target_year))
    print(file_name)
    if not os.path.exists(file_name):
        continue
    
    pred = np.load(file_name)
    tot_avg_f1.append( f1_score(pred, test_label, average="weighted") )
    tot_micro_f1.append( f1_score(pred, test_label, average="micro") )

print("average F1 %.2f $\pm$ %.2f"%(np.mean(tot_avg_f1)*100, np.std(tot_avg_f1)*100 ))
print("micro F1 %.2f $\pm$ %.2f"%(np.mean(tot_micro_f1)*100, np.std(tot_micro_f1)*100 ))

