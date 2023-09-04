##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Anurag Roy
## Modified from: https://github.com/pytorch/pytorch
## IIT Kharagpur
## anurag_roy@iitkgp.ac.in
## Copyright (c) 2023
##
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from .model_parts import *

class Tokenizer(nn.Module):
    def __init__(self,
                 kernel_size, stride, padding,
                 pooling_kernel_size=3, pooling_stride=2, pooling_padding=1,
                 n_conv_layers=1,
                 n_input_channels=3,
                 n_output_channels=64,
                 in_planes=64,
                 activation=None,
                 max_pool=True,
                 conv_bias=False,
                 split_ratio = 1.0):
        super(Tokenizer, self).__init__()
#         print(n_conv_layers)
        self.n_conv_layers = n_conv_layers
        n_filter_list = [n_input_channels] + \
                        [in_planes for _ in range(n_conv_layers - 1)] + \
                        [n_output_channels]
        self.split_ratio = split_ratio
        if split_ratio != 1.0:
            self.conv_layers = nn.ModuleList(
                [nn.Sequential(OrderedDict([
                    ("conv", Conv_SVD(n_filter_list[i], n_filter_list[i + 1],
                            kernel_size=(kernel_size, kernel_size),
                            stride=(stride, stride),
                            padding=(padding, padding), bias=conv_bias, split_ratio=split_ratio)),
                    ("activation", nn.Identity() if activation is None else activation()),
                    ("max_pool", nn.MaxPool2d(kernel_size=pooling_kernel_size,
                                stride=pooling_stride,
                                padding=pooling_padding) if max_pool else nn.Identity())
                ])
                            )
                    for i in range(n_conv_layers)
                ])
        else:
            self.conv_layers = nn.ModuleList(
                [nn.Sequential(OrderedDict([
                    ("conv", nn.Conv2d(n_filter_list[i], n_filter_list[i + 1],
                            kernel_size=(kernel_size, kernel_size),
                            stride=(stride, stride),
                            padding=(padding, padding), bias=conv_bias)),
                    ("activation", nn.Identity() if activation is None else activation()),
                    ("max_pool", nn.MaxPool2d(kernel_size=pooling_kernel_size,
                                stride=pooling_stride,
                                padding=pooling_padding) if max_pool else nn.Identity())
                ])
                            )
                    for i in range(n_conv_layers)
                ])

        self.flattener = nn.Flatten(2, 3)
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
        if self.split_ratio != 1.0:
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
        if weight_mask is None:
            for i in range(self.n_conv_layers):
                for layer in self.conv_layers[i]:
                    if isinstance(layer, Conv_SVD):
                        x = layer(x, task_id, task_ord_list)
                    else:
                        x = layer(x)
            
        else:
            idx = 0
            if bias_mask is None:               
                for layer in self.conv_layers:
                    if isinstance(layer, nn.Conv2d):
                        x = layer(x)
                        idx += 1
                    else:
                        x = layer(x)
            else:
                for layer in self.conv_layers:
                    if isinstance(layer, nn.Conv2d):
                        x = layer(x)
                        idx += 1
                    else:
                        x = layer(x)
        

        x = self.flattener(x).transpose(-2, -1)
        

        return x



    @staticmethod
    def init_weight(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight)


