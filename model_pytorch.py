import torch
import torch.nn as nn
import os
import torch.nn.functional as F
from torch.autograd import Function

class SupervisedContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07, min_tau=.07, max_tau=1., t_period=50, eps=1e-7):
    #def __init__(self, temperature=1., min_tau=.07, max_tau=1., t_period=50, eps=1e-7):
        """
        Implementation of the loss described in the paper Supervised Contrastive Learning :
        https://arxiv.org/abs/2004.11362

        :param temperature: int
        """
        super(SupervisedContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.min_tau = min_tau
        self.max_tau = max_tau
        self.t_period = t_period
        self.eps = eps

    def forward(self, projections, targets, epoch=1):
        """
        :param projections: torch.Tensor, shape [batch_size, projection_dim]
        :param targets: torch.Tensor, shape [batch_size]
        :return: torch.Tensor, scalar
        """
        device = torch.device("cuda") if projections.is_cuda else torch.device("cpu")
        

        dot_product = torch.mm(projections, projections.T)

        dot_product_tempered = dot_product / self.temperature
        
        # Minus max for numerical stability with exponential. Same done in cross entropy. Epsilon added to avoid log(0)
        stab_max, _ = torch.max(dot_product_tempered, dim=1, keepdim=True)
        exp_dot_tempered = (
            torch.exp(dot_product_tempered - stab_max.detach() ) + 1e-5
        )

        mask_similar_class = (targets.unsqueeze(1).repeat(1, targets.shape[0]) == targets).to(device)
        mask_anchor_out = (1 - torch.eye(exp_dot_tempered.shape[0])).to(device)
        mask_combined = mask_similar_class * mask_anchor_out
        cardinality_per_samples = torch.sum(mask_combined, dim=1)

        log_prob = -torch.log(exp_dot_tempered / (torch.sum(exp_dot_tempered * mask_anchor_out, dim=1, keepdim=True)))
        #### FILTER OUT POSSIBLE NaN PROBLEMS #### 
        mdf = cardinality_per_samples!=0
        cardinality_per_samples = cardinality_per_samples[mdf]
        log_prob = log_prob[mdf]
        mask_combined = mask_combined[mdf]
        #### #### #### #### #### #### #### #### #### 

        supervised_contrastive_loss_per_sample = torch.sum(log_prob * mask_combined, dim=1) / cardinality_per_samples
        supervised_contrastive_loss = torch.mean(supervised_contrastive_loss_per_sample)
        return supervised_contrastive_loss


class Conv1D_BatchNorm_Relu_Dropout(torch.nn.Module):
    def __init__(self, hidden_dims, kernel_size=5, drop_probability=0.5):
        super(Conv1D_BatchNorm_Relu_Dropout, self).__init__()

        self.block = nn.Sequential(
            nn.LazyConv1d(hidden_dims, kernel_size, padding=(kernel_size // 2)),
            nn.BatchNorm1d(hidden_dims),
            nn.ReLU(),
            nn.Dropout(p=drop_probability)
        )

    def forward(self, X):
        return self.block(X)

class FC_Classifier(torch.nn.Module):
    def __init__(self, hidden_dims, n_classes, drop_probability=0.5):
        super(FC_Classifier, self).__init__()

        self.block = nn.Sequential(
            nn.LazyLinear(hidden_dims),
            nn.BatchNorm1d(hidden_dims),
            nn.ReLU(),
            nn.Dropout(p=drop_probability),
            nn.LazyLinear(n_classes)
        )
    
    def forward(self, X):
        return self.block(X)


class TempCNN(torch.nn.Module):
    def __init__(self, num_classes=8, kernel_size=5, hidden_dims=64, dropout=0.3):
        super(TempCNN, self).__init__()
        self.conv_bn_relu1 = Conv1D_BatchNorm_Relu_Dropout(hidden_dims, kernel_size=kernel_size,
                                                           drop_probability=dropout)
        self.conv_bn_relu2 = Conv1D_BatchNorm_Relu_Dropout(hidden_dims, kernel_size=kernel_size,
                                                           drop_probability=dropout)
        self.conv_bn_relu3 = Conv1D_BatchNorm_Relu_Dropout(hidden_dims, kernel_size=kernel_size,
                                                           drop_probability=dropout)

        self.flatten = nn.Flatten()
        self.classifiers_t = FC_Classifier(256, num_classes)
        #self.discr = FC_Classifier(256, num_classes)

    def forward(self, x):
        # require NxTxD
        conv1 = self.conv_bn_relu1(x)
        conv2 = self.conv_bn_relu2(conv1)
        conv3 = self.conv_bn_relu3(conv2)
        emb = self.flatten(conv3)
        return self.classifiers_t(emb), emb 
        

class TempCNNV1(torch.nn.Module):
    def __init__(self, num_classes=8, kernel_size=5, hidden_dims=64, dropout=0.3):
        super(TempCNNV1, self).__init__()
        self.conv_bn_relu1 = Conv1D_BatchNorm_Relu_Dropout(hidden_dims, kernel_size=kernel_size,
                                                           drop_probability=dropout)
        self.conv_bn_relu2 = Conv1D_BatchNorm_Relu_Dropout(hidden_dims, kernel_size=kernel_size,
                                                           drop_probability=dropout)
        self.conv_bn_relu3 = Conv1D_BatchNorm_Relu_Dropout(hidden_dims, kernel_size=kernel_size,
                                                           drop_probability=dropout)

        self.flatten = nn.Flatten()
        #self.classifiers_t = FC_Classifier(256, num_classes)
        self.fc = nn.Sequential(
            nn.LazyLinear(256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(p=dropout)
        )
        self.classifer = nn.LazyLinear(num_classes)

    def forward(self, x):
        # require NxTxD
        conv1 = self.conv_bn_relu1(x)
        conv2 = self.conv_bn_relu2(conv1)
        conv3 = self.conv_bn_relu3(conv2)
        #emb = torch.mean(conv3,dim=2)
        emb = self.flatten(conv3)
        emb_hidden = self.flatten(conv2)
        fc_feat = self.fc(emb)
        return self.classifer(fc_feat), emb, emb_hidden, fc_feat 


class TempCNNDisentangleV4(torch.nn.Module):
    def __init__(self, num_classes=8):
        super(TempCNNDisentangleV4, self).__init__()

        self.inv = TempCNNV1(num_classes=num_classes)
        self.spec = TempCNNV1(num_classes=2)        

    def forward(self, x):
        classif, inv_emb, inv_emb_n1, inv_fc_feat = self.inv(x)
        classif_spec, spec_emb, spec_emb_n1, spec_fc_feat = self.spec(x)
        return classif, inv_emb, spec_emb, classif_spec, inv_emb_n1, spec_emb_n1, inv_fc_feat, spec_fc_feat


class TempCNNPoem(torch.nn.Module):
    def __init__(self, num_classes=8):
        super(TempCNNPoem, self).__init__()

        self.inv = TempCNNV1(num_classes=num_classes)
        self.spec = TempCNNV1(num_classes=2)
        self.classif_enc = FC_Classifier(256, 2)        

    def forward(self, x):
        classif, inv_emb, inv_emb_n1, inv_fc_feat = self.inv(x)
        classif_dom, spec_emb, spec_emb_n1, spec_fc_feat = self.spec(x)
        classif_enc = torch.cat([self.classif_enc(inv_emb),self.classif_enc(spec_emb)],dim=0)
        return classif, classif_dom, classif_enc, inv_emb, spec_emb
        
