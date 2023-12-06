import torch
import torch.nn as nn
import os
import torch.nn.functional as F
from torch.autograd import Function

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
        self.discr = FC_Classifier(256, num_classes)

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



class TempCNN_CDAN(torch.nn.Module): #TODO
    def __init__(self, size, num_classes=8, kernel_size=5, hidden_dims=64, dropout=0.3):
        super(TempCNN_CDAN, self).__init__()
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

    def forward(self, x):
        # require NxTxD
        conv1 = self.conv_bn_relu1(x)
        conv2 = self.conv_bn_relu2(conv1)
        conv3 = self.conv_bn_relu3(conv2)
        emb = self.flatten(conv3)        

        clf_output = self.classifiers_t(emb)
        softmax_output = torch.softmax(clf_output, dim=1)

        op_out = torch.bmm(softmax_output.unsqueeze(2), emb.unsqueeze(1)) # outer-product
        discr_in = op_out.view(-1, softmax_output.size(1) * emb.size(1))
        # random_out = random_layer.forward([feature, softmax_output]) #random projection (not implemented)
        # discr_in = random_out.view(-1, random_out.size(1))

        discr_out = self.discr(grad_reverse(discr_in,1.))

        return clf_output, discr_out, 0, emb



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
