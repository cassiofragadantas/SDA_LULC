# based on https://github.com/benjaminmlucas/sourcerer 
import os
import sys
sys.path.append('..')
from tqdm import tqdm
import numpy as np
import torch
import math
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import f1_score
from model_pytorch import TempCNN

class SourceRegLoss(nn.modules.module.Module):
    """
    Description:
        This is a loss function used for semi-supervised domain
        adaptation technique called "Sourerer".
        To use, a model should be trained on all labelled data from
        the source domain with a standard cross-entropy loss,
        prior to training on the available labelled target data using
        this loss.
    Args:
        source_model (pytorch model): a model trained on all labelled
                      source data.
        target_train_qty (int): the number of labelled instances
                                available in the target domain.
        target_max (int): a hyperparameter of the Sourcerer method.
                          It represents the quantity of labelled target
                          data at which the model effectively 'forgets'
                          the values of the weights learned on the
                          labelled source domain data.
    Attributes:
        lambda: the regularization constant, the amount the weights are
                'pulled' towards the values learnt on the source data.
        source_param_list: a list of the values of the parameters of
                           the model after it has been trained on the
                           source domain.
    """
    def __init__(self, source_model, target_train_qty, target_max=1e6):
        super().__init__()
        self.lambda_ = 1e10 * np.power(float(target_train_qty),
                                            (math.log(1e-20) /
                                             math.log(target_max)))

        print("Lambda value for regularization: ", self.lambda_)
        self.__module_tuple = (nn.Linear, nn.Conv1d, nn.Conv2d,
                              nn.BatchNorm1d, nn.BatchNorm2d)
        source_param_list = []
        for source_module in source_model.modules():
            if isinstance(source_module, self.__module_tuple):
                source_param_list.append(source_module.weight)
                if source_module.bias is not None:
                    source_param_list.append(source_module.bias)
        self.source_param_list = source_param_list


    def forward(self, input, target, current_model):
        """
        Description:
            Calculates the sum of the cross-entropy and SourceReg
            losses and returns the value as a pytorch tensor.
        Args:
            input (tensor): the raw logits resulting from the
                            forward-pass of the model
            target (tensor): the correct labels of the instances
            current_model (pytorch_model): a model trained on at least
                                           some labelled target data.
        Returns:
            Loss (tensor): the value of the Source-regularized loss
                           as a pytorch tensor.
        """
        cross_ent_loss = nn.CrossEntropyLoss()
        loss = cross_ent_loss(input, target)
        return self.__add_source_reg_loss(loss, current_model)


    def __add_source_reg_loss(self, loss, current_model):
        """
        (private method) Calculates the squared difference between the
        relevent parameters (weights and biases) of the current model
        and those of the source-trained model and adds this value to
        the cross-entropy loss.
        """
        current_param_list = []
        for current_module in current_model.modules():
            if isinstance(current_module, self.__module_tuple):
                current_param_list.append(current_module.weight)
                if current_module.bias is not None:
                    current_param_list.append(current_module.bias)

        for i in range(len(self.source_param_list)):
            diff = current_param_list[i].sub(self.source_param_list[i])
            sq_diff = diff.pow(2).sum()
            sq_diff_reg = self.lambda_ * sq_diff
            loss += sq_diff_reg
        return loss

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

def main():

    source_year = int(sys.argv[1])
    id_ = int(sys.argv[2])
    target_year = int(sys.argv[3])
    dataset = sys.argv[4]
    seed = int(sys.argv[5]) if len(sys.argv) > 5 else 42

    verbose = False
    epochs_source, epochs_target = 100, 100

    print(f'(Random seed set to {seed})')
    np.random.seed(seed)
    torch.manual_seed(seed)

    data_path = "../DATA_%s/"%dataset
    results_path = './'

    ##### LOAD DATA (SOURCE AND TARGET) #####
    training_batch_size = 32

    train_target_data = np.load(data_path+"train_data_%d_%d.npy"%(id_, target_year))
    train_target_label = np.load(data_path+"train_label_%d_%d.npy"%(id_, target_year))

    train_source_data = np.load(data_path+"data_%d.npy"%(source_year))
    train_source_label = np.load(data_path+"gt_data_%d.npy"%source_year)#[:,2]
    if len( train_source_label.shape) == 2:
        train_source_label = train_source_label[:,2]

    valid_data = np.load(data_path+"valid_data_%d_%d.npy"%(id_,target_year)) 
    valid_label = np.load(data_path+"valid_label_%d_%d.npy"%(id_,target_year))

    test_data = np.load(data_path+"test_data_%d_%d.npy"%(id_, target_year))
    test_label = np.load(data_path+"test_label_%d_%d.npy"%(id_, target_year))

    n_classes = len( np.unique(train_source_label))

    train_source_label = train_source_label - 1
    train_target_label = train_target_label - 1
    test_label = test_label - 1
    valid_label = valid_label - 1

    x_train_source = torch.tensor(train_source_data, dtype=torch.float32)
    y_train_source = torch.tensor(train_source_label, dtype=torch.int64)

    x_train_target = torch.tensor(train_target_data, dtype=torch.float32)
    y_train_target = torch.tensor(train_target_label, dtype=torch.int64)	
    
    # Validation dataset integrated to training data
    # x_train_target = torch.tensor(np.concatenate((train_target_data, valid_data),axis=0), dtype=torch.float32)
    # y_train_target = torch.tensor(np.concatenate((train_target_label, valid_label),axis=0), dtype=torch.int64)
    x_valid = torch.tensor(valid_data, dtype=torch.float32)
    y_valid = torch.tensor(valid_label, dtype=torch.int64)

    x_test = torch.tensor(test_data, dtype=torch.float32)
    y_test = torch.tensor(test_label, dtype=torch.int64)

    train_dataset_source = TensorDataset(x_train_source, y_train_source)
    train_dataset_target = TensorDataset(x_train_target, y_train_target)
    valid_dataset = TensorDataset(x_valid, y_valid)
    test_dataset = TensorDataset(x_test, y_test)

    source_train_generator = DataLoader(train_dataset_source, shuffle=True, batch_size=training_batch_size, drop_last=True)
    target_train_generator = DataLoader(train_dataset_target, shuffle=True, batch_size=training_batch_size, drop_last=True)
    valid_dataloader = DataLoader(valid_dataset, shuffle=False, batch_size=1024)
    target_test_generator = DataLoader(test_dataset, shuffle=False, batch_size=1024)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    ##### INITIALISE MODEL #####
    cnn = TempCNN(num_classes=n_classes).to(device)

	##### LOSS FUNCTION AND OPTIMIZER #####
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(cnn.parameters())

    model_filepath = f'{results_path}model_sourcerer_source_{dataset}_{source_year}_{id_}_{target_year}.pth'

    if os.path.isfile(model_filepath):
        print("Model Found, skipping training on Source")
        cnn = torch.load(model_filepath)
    else:
        print("Training on Source Train data...")

        ##### TRAIN MODEL #####
        cnn.train()
        
        print("No epochs: ", epochs_source)
        valid_f1 = 0.0
        for epoch in tqdm(range(epochs_source)):
            correct_val = 0
            total_val = 0
            loss_list = []
            cnn.train()
        
            for i, (X, y) in enumerate(source_train_generator):
                X = X.to(device)
                y = y.to(device)
                optimizer.zero_grad()                

                # Run the forward pass
                predictions, _ = cnn(X.float())
                loss = criterion(predictions, y)
                loss_list.append(loss.item())

                # Backprop and optimise
                loss.backward()
                optimizer.step()

                # Track the accuracy
                total_val += y.size(0)
                _, predicted = torch.max(predictions.data, dim=1)
                correct_val += (predicted == y).sum().item()

            # pred_valid, labels_valid = evaluation(cnn, valid_dataloader, device)
            # f1_val = f1_score(labels_valid, pred_valid, average="weighted")
            # if f1_val > valid_f1:
            #     torch.save(cnn, model_filepath)
            #     valid_f1 = f1_val
            #     pred_test, labels_test = evaluation(cnn, target_test_generator, device)
            #     f1 = f1_score(labels_test, pred_test, average="weighted")
            #     print("Epoch %d: F1 on TEST TARGET SET %.2f"%(epoch, 100*f1))
            # sys.stdout.flush()

        train_accuracy = correct_val/total_val
        if verbose:
            print("Train Accuracy: ", train_accuracy)

        del train_dataset_source, x_train_source, y_train_source, source_train_generator
        torch.save(cnn, model_filepath)
        # cnn = torch.load(model_filepath)

    print("Training on Target Train data (all)...")
    tfr_model_filename = f'{results_path}model_sourcerer_target_{dataset}_{source_year}_{id_}_{target_year}.pth'

    if os.path.isfile(tfr_model_filename):
        print("Model found... exiting...")
        sys.exit()

    for module in cnn.modules():
        if isinstance(module, nn.BatchNorm1d):
            module.eval()

    optimizer = optim.Adam(cnn.parameters())

    #### LOADING TARGET TRAIN DATA  #####
    target_train_qty = x_train_target.shape[0]
    print("No target training samples: ", target_train_qty)
    source_reg_loss = SourceRegLoss(cnn, target_train_qty)

    # no_updates = 5000
    # print("No updates: ", no_updates)
    # epochs_required = math.ceil(no_updates * training_batch_size / target_train_qty)
    print("No epochs: ", epochs_target)

    valid_f1 = 0.0
    for epoch in tqdm(range(epochs_target)):
        cnn.train()
        for i, (X, y) in enumerate(target_train_generator, 0):
            X = X.to(device)
            y = y.to(device)
            optimizer.zero_grad()            

            # Run the forward pass
            predictions, _ = cnn(X.float())
            loss = source_reg_loss(predictions, y, cnn)
            if verbose:
                print("Loss: ", loss.data)

            # Backprop and optimise
            loss.backward()
            optimizer.step()

        pred_valid, labels_valid = evaluation(cnn, valid_dataloader, device)
        f1_val = f1_score(labels_valid, pred_valid, average="weighted")
        if f1_val > valid_f1:
            torch.save(cnn, tfr_model_filename)
            valid_f1 = f1_val
            pred_test, labels_test = evaluation(cnn, target_test_generator, device)
            f1 = f1_score(labels_test, pred_test, average="weighted")
            print("Epoch %d: F1 on TEST TARGET SET %.2f"%(epoch, 100*f1))
        sys.stdout.flush()            

    # torch.save(cnn, tfr_model_filename)
    cnn = torch.load(tfr_model_filename)
    del train_dataset_target, x_train_target, y_train_target, target_train_generator

    ##### PREDICT TEST DATA #####
    cnn.eval()
    with torch.no_grad():
        correct_test = 0
        total_test = 0
        for i, (X, y) in enumerate(target_test_generator):
            X = X.to(device)
            y = y.to(device)

            predictions, _ = cnn(X.float())
            _, predicted = torch.max(predictions.data, dim=1)
            total_test += y.size(0)
            correct_test += (predicted == y).sum().item()

        test_accuracy = correct_test/total_test

    print("--------------------------------------------------------")
    print("Total test acc: ", test_accuracy)
    print("--------------------------------------------------------")


if __name__ == "__main__":
	main()
