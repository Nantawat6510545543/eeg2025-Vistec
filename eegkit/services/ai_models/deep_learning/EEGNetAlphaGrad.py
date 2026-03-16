import torch
import torch.nn as nn

from typing import Tuple, Optional
import torch.nn.functional as F

class Conv2dWithConstraint(nn.Conv2d):
    def __init__(self, *args, weight_norm = True, max_norm=1, **kwargs):
        self.max_norm = max_norm
        self.weight_norm = weight_norm
        super(Conv2dWithConstraint, self).__init__(*args, **kwargs)

    def forward(self, x):
        if self.weight_norm: 
            self.weight.data = torch.renorm(
                self.weight.data, p=2, dim=0, maxnorm=self.max_norm
            )
        return super(Conv2dWithConstraint, self).forward(x)
    
class SeparableConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, padding=1, bias=False, **kwargs):
        super(SeparableConv2d, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, 
                                groups=in_channels, bias=bias, padding=padding, **kwargs)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 
                                kernel_size=1, bias=bias, **kwargs)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out

class LinearWithConstraint(nn.Linear):
    def __init__(self, *args, weight_norm = True, max_norm=1, **kwargs):
        self.max_norm = max_norm
        self.weight_norm = weight_norm
        super(LinearWithConstraint, self).__init__(*args, **kwargs)

    def forward(self, x):
        if self.weight_norm: 
            self.weight.data = torch.renorm(
                self.weight.data, p=2, dim=0, maxnorm=self.max_norm
            )
        return super(LinearWithConstraint, self).forward(x)

class EEGNet(nn.Module):
    
    def __init__(self, n_channel, n_time, n_class=2,
                 dropout=0.5, F1=8, D=2,
                 C1=200, *args, **kwargs):
        super(EEGNet, self).__init__()
        self.F2 = D*F1
        self.F1 = F1
        self.D = D
        self.n_time = n_time
        self.n_class = n_class
        self.n_channel = n_channel
        self.C1 = C1

        self.conv_layer = self.initial_block(dropout)
        self.flatten_size = self.cal_flat_size(self.conv_layer)
        self.flatten_layer = self.flatten_block(self.flatten_size, n_class)
        
    def initial_block(self, dropout, *args, **kwargs):
        block1 = nn.Sequential(
                nn.Conv2d(1, self.F1, (1, self.C1),
                          padding='same', bias=False),
                nn.BatchNorm2d(self.F1),
                Conv2dWithConstraint(self.F1, self.F1*self.D, (self.n_channel, 1),
                                     padding=0, bias=False, max_norm=1,
                                     groups=self.D),
                nn.BatchNorm2d(self.F1*self.D),
                nn.ELU(),
                nn.AvgPool2d((1,4)),
                nn.Dropout(p=dropout),
                SeparableConv2d(self.F1*self.D, self.F2, (1, self.C1//4), padding='same', bias=False),
                nn.BatchNorm2d(self.F2),
                nn.ELU(),
                nn.AvgPool2d((1,8)),
                nn.Dropout(p=dropout)
                )
        return block1

    def flatten_block(self, in_features, out_features, *args, **kwargs):
        return nn.Sequential(
            nn.Flatten(),
            LinearWithConstraint(
                in_features=in_features,
                out_features=out_features,
                max_norm=0.25,
                weight_norm=True,
            ),
            nn.Softmax(dim=1),
        )

    def cal_flat_size(self, model):
        '''
        Calculate the output based on input size.
        model is from nn.Module.
        '''
        data = torch.rand(1,1,self.n_channel, self.n_time)
        model.eval()
        out = model(data).shape
        return int(torch.Tensor([out[1:]]).prod().item())

    def forward(self, x):
        x = self.conv_layer(x)
        x = self.flatten_layer(x)
        return x
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)