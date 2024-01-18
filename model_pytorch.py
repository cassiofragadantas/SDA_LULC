import torch
import torch.nn as nn
import os
import torch.nn.functional as F
from torch.autograd import Function

#####TEMPERATURE ANNEALING######
'''
      if args.adj_tau == 'cos':
            t_max = args.t_max
            min_tau = args.temperature_min
            max_tau = args.temperature_max
            
            temperature = min_tau + 0.5 * (max_tau - min_tau) * (1 + torch.cos(torch.tensor(torch.pi * epoch / self.t_period )))

'''
##################################

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
        #temperature = self.min_tau + 0.5 * (self.max_tau - self.min_tau) * (1 + torch.cos(torch.tensor(torch.pi * epoch / self.t_period )))
        

        dot_product = torch.mm(projections, projections.T)
        ### For stability issues related to matrix multiplications
        #dot_product = torch.clamp(dot_product, -1+self.eps, 1-self.eps)
        ####GEODESIC SIMILARITY
        #print(projections)
        #print( dot_product )
        #print( torch.acos(dot_product) / torch.pi )
        #dot_product = 1. - ( torch.acos(dot_product) / torch.pi )

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
        #print(log_prob)
        #### FILTER OUT POSSIBLE NaN PROBLEMS #### 
        mdf = cardinality_per_samples!=0
        cardinality_per_samples = cardinality_per_samples[mdf]
        log_prob = log_prob[mdf]
        mask_combined = mask_combined[mdf]
        #### #### #### #### #### #### #### #### #### 

        supervised_contrastive_loss_per_sample = torch.sum(log_prob * mask_combined, dim=1) / cardinality_per_samples
        #print(supervised_contrastive_loss_per_sample)
        supervised_contrastive_loss = torch.mean(supervised_contrastive_loss_per_sample)
        #print(supervised_contrastive_loss)
        #print("============")
        return supervised_contrastive_loss






def Entropy(input_):
    bs = input_.size(0)
    epsilon = 1e-5
    entropy = -input_ * torch.log(input_ + epsilon)
    entropy = torch.sum(entropy, dim=1)
    return entropy


class GradReverse(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)
        #print(alpha)
    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output * -ctx.alpha
        return output, None

def grad_reverse(x,alpha):
    return GradReverse.apply(x,alpha)


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
        return self.classifiers_t(emb), emb #self.discr(grad_reverse(emb,1.)), emb
        
class TempCNNWP(torch.nn.Module):
    def __init__(self, size, proj_dim=64, num_classes=8, kernel_size=5, hidden_dims=64, dropout=0.3):
        super(TempCNNWP, self).__init__()
        #self.modelname = f"TempCNN_input-dim={input_dim}_num-classes={num_classes}_sequencelenght={sequencelength}_" \
        #                 f"kernelsize={kernel_size}_hidden-dims={hidden_dims}_dropout={dropout}"
        self.nchannels = size[0]
        self.nts = size[1]

        self.hidden_dims = hidden_dims

        self.conv_bn_relu1 = Conv1D_BatchNorm_Relu_Dropout(hidden_dims, kernel_size=kernel_size,
                                                           drop_probability=dropout)
        self.conv_bn_relu2 = Conv1D_BatchNorm_Relu_Dropout(hidden_dims, kernel_size=kernel_size,
                                                           drop_probability=dropout)
        self.conv_bn_relu3 = Conv1D_BatchNorm_Relu_Dropout(hidden_dims, kernel_size=kernel_size,
                                                           drop_probability=dropout)
        self.flatten = nn.Flatten()
        self.classifiers_t = FC_Classifier(256, num_classes)
        self.discr = FC_Classifier(256, 2)

        self.reco = nn.LazyLinear(self.nchannels * self.nts)

        self.proj_head = nn.LazyLinear(proj_dim)

    def forward(self, x):
        # require NxTxD
        conv1 = self.conv_bn_relu1(x)
        conv2 = self.conv_bn_relu2(conv1)
        conv3 = self.conv_bn_relu3(conv2)
        emb = self.flatten(conv3)        
        ##############################
        reco = self.reco(emb)
        reco = reco.view(-1,self.nchannels,self.nts)

        return self.classifiers_t(emb), self.discr(grad_reverse(emb,1.)), reco, emb


class TempCNNWP2(torch.nn.Module):
    def __init__(self, size, num_classes=8, kernel_size=5, hidden_dims=64, dropout=0.3):
        super(TempCNNWP2, self).__init__()
        #self.modelname = f"TempCNN_input-dim={input_dim}_num-classes={num_classes}_sequencelenght={sequencelength}_" \
        #                 f"kernelsize={kernel_size}_hidden-dims={hidden_dims}_dropout={dropout}"
        self.nchannels = size[0]
        self.nts = size[1]

        self.hidden_dims = hidden_dims

        self.conv_bn_relu1 = Conv1D_BatchNorm_Relu_Dropout(hidden_dims, kernel_size=kernel_size,
                                                           drop_probability=dropout)
        self.conv_bn_relu2 = Conv1D_BatchNorm_Relu_Dropout(hidden_dims, kernel_size=kernel_size,
                                                           drop_probability=dropout)
        self.conv_bn_relu3 = Conv1D_BatchNorm_Relu_Dropout(hidden_dims, kernel_size=kernel_size,
                                                           drop_probability=dropout)


        self.conv_bn_relu1_sets = nn.ModuleList( [Conv1D_BatchNorm_Relu_Dropout(hidden_dims, kernel_size=kernel_size, drop_probability=dropout) for _ in range(num_classes)] )
        self.conv_bn_relu2_sets = nn.ModuleList( [Conv1D_BatchNorm_Relu_Dropout(hidden_dims, kernel_size=kernel_size, drop_probability=dropout) for _ in range(num_classes)] )
        self.conv_bn_relu3_sets = nn.ModuleList( [Conv1D_BatchNorm_Relu_Dropout(hidden_dims, kernel_size=kernel_size, drop_probability=dropout) for _ in range(num_classes)] )
        self.flatten = nn.Flatten()
        self.classifiers_t = FC_Classifier(256, num_classes)
        self.discr = FC_Classifier(256, num_classes)

        self.reco = nn.LazyLinear(self.nchannels * self.nts)

    def forward(self, x):
        # require NxTxD
        conv1 = self.conv_bn_relu1(x)
        conv2 = self.conv_bn_relu2(conv1)
        conv3 = self.conv_bn_relu3(conv2)
        emb = self.flatten(conv3)
        reco = self.reco(emb)
        reco = reco.view(-1,self.nchannels,self.nts)

        return self.classifiers_t(emb), self.discr(grad_reverse(emb,1.)), reco, emb


class TempCNNWP3(torch.nn.Module):
    def __init__(self, size, proj_head_dim, num_classes=8, kernel_size=5, hidden_dims=64, dropout=0.3):
        super(TempCNNWP3, self).__init__()
        #self.modelname = f"TempCNN_input-dim={input_dim}_num-classes={num_classes}_sequencelenght={sequencelength}_" \
        #                 f"kernelsize={kernel_size}_hidden-dims={hidden_dims}_dropout={dropout}"
        self.nchannels = size[0]
        self.nts = size[1]

        self.hidden_dims = hidden_dims

        self.conv_bn_relu1 = Conv1D_BatchNorm_Relu_Dropout(hidden_dims, kernel_size=kernel_size,
                                                           drop_probability=dropout)
        self.conv_bn_relu2 = Conv1D_BatchNorm_Relu_Dropout(hidden_dims, kernel_size=kernel_size,
                                                           drop_probability=dropout)
        self.conv_bn_relu3 = Conv1D_BatchNorm_Relu_Dropout(hidden_dims, kernel_size=kernel_size,
                                                           drop_probability=dropout)


        self.conv_bn_relu1_sets = nn.ModuleList( [Conv1D_BatchNorm_Relu_Dropout(hidden_dims, kernel_size=kernel_size, drop_probability=dropout) for _ in range(num_classes)] )
        self.conv_bn_relu2_sets = nn.ModuleList( [Conv1D_BatchNorm_Relu_Dropout(hidden_dims, kernel_size=kernel_size, drop_probability=dropout) for _ in range(num_classes)] )
        self.conv_bn_relu3_sets = nn.ModuleList( [Conv1D_BatchNorm_Relu_Dropout(hidden_dims, kernel_size=kernel_size, drop_probability=dropout) for _ in range(num_classes)] )
        self.flatten = nn.Flatten()
        self.classifiers_t = FC_Classifier(256, num_classes)
        self.discr = FC_Classifier(256, 2)
        self.head = nn.LazyLinear(proj_head_dim)
        self.head2 = nn.LazyLinear(proj_head_dim)

        self.reco = nn.LazyLinear(self.nchannels * self.nts)

    def forward(self, x):
        # require NxTxD
        conv1 = self.conv_bn_relu1(x)
        conv2 = self.conv_bn_relu2(conv1)
        conv3 = self.conv_bn_relu3(conv2)
        emb = self.flatten(conv3)
        reco = self.reco(emb)
        reco = reco.view(-1,self.nchannels,self.nts)

        head_proj = self.head(emb)

        return self.classifiers_t(emb), self.head2(grad_reverse(emb,1.)), reco, head_proj





################################################################################
class InceptionLayer(nn.Module):
    # PyTorch translation of the Keras code in https://github.com/hfawaz/dl-4-tsc
    def __init__(self, nb_filters=32, use_bottleneck=True,
                 bottleneck_size=32, kernel_size=40):
        super(InceptionLayer, self).__init__()

        # self.in_channels = in_channels
        kernel_size_s = [(kernel_size) // (2 ** i) for i in range(3)] # = [40, 20, 10]
        kernel_size_s = [x+1 for x in kernel_size_s] # Avoids warning about even kernel_size with padding="same"
        self.bottleneck_size = bottleneck_size
        self.use_bottleneck = use_bottleneck


        # Bottleneck layer
        self.bottleneck = nn.LazyConv1d(self.bottleneck_size, kernel_size=1,
                                    stride=1, padding="same", bias=False)
        self.max_pool = nn.MaxPool1d(kernel_size=3, stride=1, padding=1)
        self.bottleneck_conv = nn.LazyConv1d(nb_filters, kernel_size=1,
                                         stride=1, padding="same", bias=False)

        # Convolutional layer (several filter lenghts)
        self.conv_list = nn.ModuleList([])
        for i in range(len(kernel_size_s)):
            # Input size could be self.in_channels or self.bottleneck_size (if bottleneck was applied)
            self.conv_list.append(nn.LazyConv1d(nb_filters, kernel_size=kernel_size_s[i],
                                            stride=1, padding='same', bias=False))

        self.bn = nn.BatchNorm1d(4*self.bottleneck_size)
        self.relu = nn.ReLU()

    def forward(self, input):
        in_channels = input.shape[-2]
        if self.use_bottleneck and int(in_channels) > self.bottleneck_size:
            input_inception = self.bottleneck(input)
        else:
            input_inception = input

        max_pool = self.max_pool(input)
        output = self.bottleneck_conv(max_pool)
        for conv in self.conv_list:
            output = torch.cat((output,conv(input_inception)),dim=1)

        output = self.bn(output)
        output = self.relu(output)

        return output

'''
class InceptionBranch(nn.Module):
    # PyTorch translation of the Keras code in https://github.com/hfawaz/dl-4-tsc
    def __init__(self, nb_filters=32, use_residual=True,
                 use_bottleneck=True, bottleneck_size=32, depth=6, kernel_size=41):
        super(InceptionBranch, self).__init__()

        self.use_residual = use_residual

        # Inception layers
        self.inception_list = nn.ModuleList(
            [InceptionLayer(nb_filters,use_bottleneck, bottleneck_size, kernel_size) for _ in range(depth)])
        # Explicit input sizes (i.e. without using Lazy layers). Requires n_var passed as a constructor input
        # self.inception_list = nn.ModuleList([InceptionLayer(n_var, nb_filters,use_bottleneck, bottleneck_size, kernel_size) for _ in range(depth)])
        # for _ in range(1,depth):
        #     inception = InceptionLayer(4*nb_filters,nb_filters,use_bottleneck, bottleneck_size, kernel_size)
        #     self.inception_list.append(inception)

        # Fully-connected layer
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.out = nn.Flatten()

        # Shortcut layers
        # First residual layer has n_var channels as inputs while the remaining have 4*nb_filters
        self.conv = nn.ModuleList([
            nn.LazyConv1d(4*nb_filters, kernel_size=1,
                            stride=1, padding="same", bias=False)
            for _ in range(int(depth/3))
        ])
        self.bn = nn.ModuleList([nn.BatchNorm1d(4*nb_filters) for _ in range(int(depth/3))])
        self.relu = nn.ModuleList([nn.ReLU() for _ in range(int(depth/3))])

    def _shortcut_layer(self, input_tensor, out_tensor, id):
        shortcut_y = self.conv[id](input_tensor)
        shortcut_y = self.bn[id](shortcut_y)
        x = torch.add(shortcut_y, out_tensor)
        x = self.relu[id](x)
        return x

    def forward(self, x):
        input_res = x

        for d, inception in enumerate(self.inception_list):
            x = inception(x)

            # Residual layer
            if self.use_residual and d % 3 == 2:
                x = self._shortcut_layer(input_res,x, int(d/3))
                input_res = x

        gap_layer = self.gap(x)
        return self.out(gap_layer)
'''

class Inception(nn.Module):
    # PyTorch translation of the Keras code in https://github.com/hfawaz/dl-4-tsc
    def __init__(self, nb_classes, nb_filters=32, use_residual=True,
                 use_bottleneck=True, bottleneck_size=32, depth=6, kernel_size=41):
        super(Inception, self).__init__()

        self.use_residual = use_residual

        # Inception layers
        self.inception_list = nn.ModuleList(
            [InceptionLayer(nb_filters,use_bottleneck, bottleneck_size, kernel_size) for _ in range(depth)])
        # Explicit input sizes (i.e. without using Lazy layers). Requires n_var passed as a constructor input
        # self.inception_list = nn.ModuleList([InceptionLayer(n_var, nb_filters,use_bottleneck, bottleneck_size, kernel_size) for _ in range(depth)])
        # for _ in range(1,depth):
        #     inception = InceptionLayer(4*nb_filters,nb_filters,use_bottleneck, bottleneck_size, kernel_size)
        #     self.inception_list.append(inception)

        # Fully-connected layer
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.LazyLinear(nb_classes),
            # nn.Softmax(dim=1) # already performed inside CrossEntropyLoss
        )

        # Shortcut layers
        # First residual layer has n_var channels as inputs while the remaining have 4*nb_filters
        self.conv = nn.ModuleList([
            nn.LazyConv1d(4*nb_filters, kernel_size=1,
                            stride=1, padding="same", bias=False)
            for _ in range(int(depth/3))
        ])
        self.bn = nn.ModuleList([nn.BatchNorm1d(4*nb_filters) for _ in range(int(depth/3))])
        self.relu = nn.ModuleList([nn.ReLU() for _ in range(int(depth/3))])

    def _shortcut_layer(self, input_tensor, out_tensor, id):
        shortcut_y = self.conv[id](input_tensor)
        shortcut_y = self.bn[id](shortcut_y)
        x = torch.add(shortcut_y, out_tensor)
        x = self.relu[id](x)
        return x

    def forward(self, x):
        input_res = x

        for d, inception in enumerate(self.inception_list):
            x = inception(x)

            # Residual layer
            if self.use_residual and d % 3 == 2:
                x = self._shortcut_layer(input_res,x, int(d/3))
                input_res = x

        gap_layer = self.gap(x)
        return self.fc(gap_layer), gap_layer


class Reco(torch.nn.Module):
    def __init__(self, ts_length, n_bands, drop_probability=0.5):
        super(Reco, self).__init__()

        self.final_dim = ts_length * n_bands
        self.ts_length = ts_length
        self.n_bands = n_bands

        self.reco1 = nn.LazyLinear(self.final_dim // 2 )
        self.bn1 = nn.BatchNorm1d(self.final_dim // 2)
        self.dp = nn.Dropout(p=drop_probability)
        self.reco2 = nn.LazyLinear(self.final_dim)
    
    def forward(self, x):
        reco = self.reco1(x)
        reco = self.bn1(reco)
        reco = self.dp(reco)
        reco = self.reco2(reco)
        #reco = self.reco2(x)
        reco = reco.view(-1,self.n_bands,self.ts_length)
        return reco

class TempCNNReco(torch.nn.Module):
    def __init__(self, ts_length, n_bands, num_classes=8, kernel_size=5, hidden_dims=64, dropout=0.3):
        super(TempCNNReco, self).__init__()
        self.conv_bn_relu1 = Conv1D_BatchNorm_Relu_Dropout(hidden_dims, kernel_size=kernel_size,
                                                           drop_probability=dropout)
        self.conv_bn_relu2 = Conv1D_BatchNorm_Relu_Dropout(hidden_dims, kernel_size=kernel_size,
                                                           drop_probability=dropout)
        self.conv_bn_relu3 = Conv1D_BatchNorm_Relu_Dropout(hidden_dims, kernel_size=kernel_size,
                                                           drop_probability=dropout)

        self.flatten = nn.Flatten()
        self.classifiers_t = FC_Classifier(256, num_classes)
        self.reco = Reco(ts_length, n_bands)

    def forward(self, x):
        # require NxTxD
        conv1 = self.conv_bn_relu1(x)
        conv2 = self.conv_bn_relu2(conv1)
        conv3 = self.conv_bn_relu3(conv2)
        emb = self.flatten(conv3)
        return self.classifiers_t(emb), emb, self.reco(emb) #self.discr(grad_reverse(emb,1.)), emb
    

class ProjHead(torch.nn.Module):
    def __init__(self, projDim, drop_probability=0.2):
        super(ProjHead, self).__init__()

        self.proj1 = nn.LazyLinear(projDim)    
        self.bn1 = nn.BatchNorm1d(projDim)
        self.relu1 = nn.ReLU()
        self.dp1 = nn.Dropout(p=drop_probability)
        #self.proj2 = nn.LazyLinear(projDim)
        #self.relu2 = nn.ReLU()
        #self.dp2 = nn.Dropout(p=drop_probability)
    
    def forward(self, x):
        emb = self.proj1(x)
        emb = self.bn1(emb)
        emb = self.relu1(emb)
        emb = self.dp1(emb)
        #emb = self.proj2(emb)
        #emb = self.dp2(emb) 

        return emb


class TempCNNDisentangle(torch.nn.Module):
    def __init__(self, ts_length, n_bands, num_classes=8, kernel_size=5, hidden_dims=64, dropout=0.5):
        super(TempCNNDisentangle, self).__init__()

        self.enc = nn.Sequential(
            Conv1D_BatchNorm_Relu_Dropout(hidden_dims, kernel_size=kernel_size,
                                                           drop_probability=dropout),
            Conv1D_BatchNorm_Relu_Dropout(hidden_dims, kernel_size=kernel_size,
                                                           drop_probability=dropout),
            Conv1D_BatchNorm_Relu_Dropout(hidden_dims, kernel_size=kernel_size,
                                                           drop_probability=dropout)
        )

        '''
        self.conv_bn_relu1 = Conv1D_BatchNorm_Relu_Dropout(hidden_dims, kernel_size=kernel_size,
                                                           drop_probability=dropout)
        self.conv_bn_relu2 = Conv1D_BatchNorm_Relu_Dropout(hidden_dims, kernel_size=kernel_size,
                                                           drop_probability=dropout)
        self.conv_bn_relu3 = Conv1D_BatchNorm_Relu_Dropout(hidden_dims, kernel_size=kernel_size,
                                                           drop_probability=dropout)
        '''

        self.flatten = nn.Flatten()
        self.classifiers_t = FC_Classifier(256, num_classes)
        self.classifiers_d_inv = FC_Classifier(256, 2)
        self.reco = Reco(ts_length, n_bands)
        
        self.classifiers_d_spec = FC_Classifier(256, 2)
        
        '''
        self.gating = nn.Sequential(
            nn.LazyLinear(4672),
            nn.Sigmoid()
        )
       
        
        self.inv_f = ProjHead(256)
        self.spec_f = ProjHead(256)
        '''
        self.proj = ProjHead(128)

        self.compress = nn.Sequential(
            nn.LazyLinear(1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(p=dropout)
        )
        self.softm = nn.Softmax(dim=1)

    def forward(self, x, alpha=1.0):
        # require NxTxD
        #conv1 = self.conv_bn_relu1(x)
        #conv2 = self.conv_bn_relu2(conv1)
        #conv3 = self.conv_bn_relu3(conv2)
        conv3 = self.enc(x)
        #emb = torch.mean(conv3,dim=2)
        
        #print(emb.shape)
        #exit()
        emb = self.flatten(conv3)
        emb = self.proj(emb)
        #emb = self.compress(emb)
        inv_emb = emb[:,emb.shape[1]//2:]
        spec_emb = emb[:,:emb.shape[1]//2]        

        classif = self.classifiers_t(inv_emb)
        
        '''DANN'''
        domain_classif = self.classifiers_d_inv(grad_reverse(inv_emb,alpha))
        ''' ''' 

        ''' CDAN '''
        '''
        softmax_output = self.softm( classif.detach() )
        cdan_features = torch.bmm(softmax_output.unsqueeze(2), inv_emb.unsqueeze(1))
        cdan_features = cdan_features.view(-1, softmax_output.size(1) * inv_emb.size(1))
        domain_classif = self.classifiers_d_inv(grad_reverse(cdan_features,alpha))
        '''
        
        ''' ''' 

        return classif, inv_emb, spec_emb, self.reco(emb), domain_classif, self.classifiers_d_spec(spec_emb)  #self.discr(grad_reverse(emb,1.)), emb


class TempCNNDisentangleV2(torch.nn.Module):
    def __init__(self, ts_length, n_bands, num_classes=8, kernel_size=5, hidden_dims=64, dropout=0.5):
        super(TempCNNDisentangleV2, self).__init__()

        self.enc = nn.Sequential(
            Conv1D_BatchNorm_Relu_Dropout(hidden_dims, kernel_size=kernel_size,
                                                           drop_probability=dropout),
            Conv1D_BatchNorm_Relu_Dropout(hidden_dims, kernel_size=kernel_size,
                                                           drop_probability=dropout),
            Conv1D_BatchNorm_Relu_Dropout(hidden_dims, kernel_size=kernel_size,
                                                           drop_probability=dropout)
        )

        self.flatten = nn.Flatten()
        self.classifiers_t = FC_Classifier(256, num_classes)
        
        self.classifiers_d_inv = FC_Classifier(256, 2)
        
        
        #self.reco = Reco(ts_length, n_bands)
        '''
        self.classifiers_d_spec = FC_Classifier(256, 2)
        self.classifiers_dc_spec = FC_Classifier(256, 2*num_classes)

        '''
        self.classifiers_d_spec = nn.LazyLinear(2)
        self.classifiers_dc_spec = nn.LazyLinear(2*num_classes)        
        
        
        #self.proj = ProjHead(192)
        self.proj2 = ProjHead(256)
        self.classif = nn.LazyLinear(num_classes)
        #self.softm = nn.Softmax(dim=1)

    def forward(self, x, alpha=1.0):
        '''
        conv3 = self.enc(x)
        emb = self.flatten(conv3)
        emb = self.proj(emb)
        step = emb.shape[1]//3
        inv_emb = emb[:,:step]
        spec_emb_dc = emb[:,step:2*step]        
        spec_emb_d = emb[:,2*step::]
        classif = self.classifiers_t(inv_emb)

        '''
        conv3 = self.enc(x)
        emb = self.flatten(conv3)
        emb = self.proj2(emb)
        inv_emb = emb[:,emb.shape[1]//2:]
        spec_emb = emb[:,:emb.shape[1]//2]
        classif = self.classif(inv_emb)
        
        
        '''DANN'''
        #domain_classif = self.classifiers_d_inv(grad_reverse(inv_emb,alpha))
        ''' ''' 

        ''' CDAN '''
        '''
        softmax_output = self.softm( classif.detach() )
        cdan_features = torch.bmm(softmax_output.unsqueeze(2), inv_emb.unsqueeze(1))
        cdan_features = cdan_features.view(-1, softmax_output.size(1) * inv_emb.size(1))
        domain_classif = self.classifiers_d_inv(grad_reverse(cdan_features,alpha))
        '''
        
        ''' ''' 

        return classif, inv_emb, spec_emb, self.classifiers_d_spec(spec_emb) #self.classifiers_dc_spec(spec_emb) #

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
        return self.classifer(fc_feat), emb, emb_hidden, fc_feat #self.discr(grad_reverse(emb,1.)), emb


class TempCNNDisentangleV3(torch.nn.Module):
    def __init__(self, num_classes=8):
        super(TempCNNDisentangleV3, self).__init__()

        self.inv = TempCNNV1(num_classes=num_classes)
        self.spec = TempCNNV1(num_classes=2)        

    def forward(self, x):
        classif, inv_emb, _, _ = self.inv(x)
        classif_spec, spec_emb, _, _ = self.spec(x)
        return classif, inv_emb, spec_emb, classif_spec

class TempCNNDisentangleV3(torch.nn.Module):
    def __init__(self, num_classes=8):
        super(TempCNNDisentangleV3, self).__init__()

        self.inv = TempCNNV1(num_classes=num_classes)
        self.spec = TempCNNV1(num_classes=2)        

    def forward(self, x):
        classif, inv_emb, inv_emb_n1, inv_fc_feat = self.inv(x)
        classif_spec, spec_emb, spec_emb_n1, spec_fc_feat = self.spec(x)
        return classif, inv_emb, spec_emb, classif_spec, inv_emb_n1, spec_emb_n1, inv_fc_feat, spec_fc_feat

    
class InceptionDisentangle(torch.nn.Module):
    def __init__(self, ts_length, n_bands, num_classes=8, kernel_size=5, hidden_dims=64, dropout=0.5):
        super(InceptionDisentangle, self).__init__()

        self.enc = Inception(num_classes) #nb_filters=16 # output embedding is of size 4*nb_filters

        self.flatten = nn.Flatten()
        self.classifiers_t = FC_Classifier(256, num_classes)
        self.classifiers_d_inv = FC_Classifier(256, 2)
        self.reco = Reco(ts_length, n_bands)

        self.classifiers_d_spec = FC_Classifier(256, 2)

    def forward(self, x):
        _, _, conv3 = self.enc(x) # before GAP layer. Shape (N, 128, T)
        # _, conv3, _ = self.enc(x) # after GAP layer. Shape (N, 128, 1)
        emb = self.flatten(conv3)
        inv_emb = emb[:,emb.shape[1]//2:]
        spec_emb = emb[:,:emb.shape[1]//2]
        return self.classifiers_t(inv_emb), inv_emb, spec_emb, self.reco(emb), self.classifiers_d_inv(grad_reverse(inv_emb,1.)), self.classifiers_d_spec(spec_emb)
