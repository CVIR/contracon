##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Anurag Roy
## Modified from: https://github.com/pytorch/pytorch
## IIT Kharagpur
## anurag_roy@iitkgp.ac.in
## Copyright (c) 2021
##
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

from lib2to3.pgen2.tokenize import Untokenizer
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from .model_parts import *
import copy

class UnTokenizer(nn.Module):
    def __init__(self,
                 kernel_size, stride, padding,
                 pooling_kernel_size=2, pooling_stride=2, pooling_padding=1,
                 n_conv_layers=1,
                 n_input_channels=3,
                 n_output_channels=64,
                 in_planes=64,
                 activation=None,
                 max_pool=True,
                 conv_bias=False,
                 split_ratio = 1.0):
        super(UnTokenizer, self).__init__()
        self.n_conv_layers = n_conv_layers
        n_filter_list = [n_input_channels] + \
                        [in_planes for _ in range(n_conv_layers - 1)] + \
                        [n_output_channels]
        n_filter_list = n_filter_list[::-1]
        self.split_ratio = split_ratio

        self.conv_layers = nn.ModuleList(
            [nn.Sequential(OrderedDict([
                ("conv", ConvTranspose2d_SVD(n_filter_list[i], n_filter_list[i + 1],
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=padding, bias=conv_bias)),
                ("activation", nn.Identity() if activation is None else activation()),
                ("max_pool", nn.Upsample(scale_factor=pooling_kernel_size, mode='bilinear'))
            ])
                        )
                for i in range(n_conv_layers)
            ])
        self.unflattener = torch.nn.Unflatten(2, torch.Size([ 16, 16]))
        self.parent_task = True

        self.apply(self.init_weight)

    def set_parent(self, val):
        self.parent_task = val
        if self.split_ratio != 1.0:
            for i in range(self.n_conv_layers):
                self.conv_layers[i][0].set_parent(val)

    def reset_parameters(self):
        nn.init.ones_(self.conv_scale)
        nn.init.ones_(self.conv_shift)

    def update_global(self, task_id):
        for i in range(self.n_conv_layers):
            self.conv_layers[i][0].update_global(task_id)        
            
    def save_params(self, save_path):
        if self.split_ratio != 1.0:
            for i in range(self.n_conv_layers):
                self.conv_layers[i][0].save_params('{}_layer{}'.format(save_path, 0))

    def load_params(self, load_path):
        if self.split_ratio != 1.0:
            for i in range(self.n_conv_layers):
                self.conv_layers[i][0].load_params('{}_layer{}'.format(load_path, 0))

    def set_similar_task(self, task_id):
        if self.split_ratio != 1.0:
            for i in range(self.n_conv_layers):
                self.conv_layers[i][0].set_similar_task(task_id)
    
    def sequence_length(self, n_channels=3, height=224, width=224):
        return self.forward(torch.zeros((1, n_channels, height, width))).shape[1]

    def forward(self, x, task_id=-1, task_ord_list = [], weight_mask=None, bias_mask=None):
        x = x.transpose(-2, -1)
        x = self.unflattener(x)
        
        for i in range(self.n_conv_layers):
            for layer in self.conv_layers[i]:
                if isinstance(layer, ConvTranspose2d_SVD):
                    x = layer(x, task_id, task_ord_list)
                else:
                    x = layer(x)

        return x



    @staticmethod
    def init_weight(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight)


