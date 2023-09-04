import torch
import numbers
from torch import nn
from torch.nn import functional as F
from torch.nn import Module, ModuleList, Linear, Dropout, LayerNorm, Identity, Parameter, init
from torch.autograd import Variable
import torch.utils.data
import torch.utils.data.distributed
from torchvision import utils
import copy
import pickle as pkl
import math
import os
import numpy as np
import matplotlib.pyplot as plt

def my_copy(x):
    return copy.deepcopy(x)

def visTensor(tensor, fname, ch=0, allkernels=False, nrow=8, padding=1): 
    n,c,w,h = tensor.shape

    if allkernels: tensor = tensor.view(n*c, -1, w, h)
    elif c != 3: tensor = tensor[:,ch,:,:].unsqueeze(dim=1)

    rows = np.min((tensor.shape[0] // nrow + 1, nrow * nrow))    
    grid = utils.make_grid(tensor, nrow=nrow, normalize=True, padding=padding)
    plt.figure( figsize=(nrow,rows) )
    plt.imsave(fname, grid.numpy().transpose((1, 2, 0)))
    plt.close()
#################################FCBLock SVD#############################################
class FCBlock_SVD(nn.Module):
    def __init__(self, fin, fout, bias=True):
        super(FCBlock_SVD, self).__init__()
        self.fc = nn.Linear(fin, fout, bias=bias)
        self.global_fcblk = {}  
        self.fc.reset_parameters()
            
    def forward(self, inp, task_id=-1, task_ord_list=[]):
        x = inp
        if task_id == -1:
            return self.fc(x)
        else:
            fc = self.global_fcblk[task_id]
            return fc(x)
        


    def update_global(self, task_id):
        # self.fc.eval()
        self.global_fcblk[task_id] = copy.deepcopy(self.fc)
        # self.fc.train()
        self.fc.reset_parameters()

        
    def get_task_params(self, prefix, task_id):
        fc_params = []
        if task_id == -1:
            fc = self.fc
        else:
            fc = self.global_fcblk[task_id]
        
        for name, param  in fc.named_parameters():
            fc_params.append(("{}.{}.fc.{}".format(prefix, name, task_id), param))
        
        return fc_params
    

    def save_params(self, save_path):
        with open("{}_global_fcblk.pkl".format(save_path), 'wb') as fid:
            pkl.dump(self.global_fcblk, fid)


           
    def load_params(self, load_path):
        with open("{}_global_fcblk.pkl".format(load_path), 'rb') as fid:
            self.global_fcblk = pkl.load(fid)

    def set_similar_task(self, task_id):
        self.fc.weight = copy.deepcopy(self.global_fcblk[task_id].weight)
        if self.fc.bias is not None:
            self.fc.bias = copy.deepcopy(self.global_fcblk[task_id].bias)



#################################Conv1d SVD#############################################
class Conv2D_SVD(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super(Conv2D_SVD, self).__init__()
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)
        self.global_cnv1d = {}  
        self.conv2d.reset_parameters()
            
    def forward(self, inp, task_id=-1, task_ord_list=[]):
        x = inp
        if task_id == -1:
            return self.conv2d(x)
        else:
            conv1d = self.global_cnv1d[task_id]
            return conv1d(x)
        


    def update_global(self, task_id):
        self.global_cnv1d[task_id] = copy.deepcopy(self.conv2d)
        self.conv2d.reset_parameters()

        
    def get_task_params(self, prefix, task_id):
        conv1d_params = []
        if task_id == -1:
            conv1d = self.conv2d
        else:
            conv1d = self.global_cnv1d[task_id]
        
        for name, param  in conv1d.named_parameters():
            conv1d_params.append(("{}.{}.conv1d.{}".format(prefix, name, task_id), param))
        
        return conv1d_params
    

    def save_params(self, save_path):
        with open("{}_global_cnv2d.pkl".format(save_path), 'wb') as fid:
            pkl.dump(self.global_cnv1d, fid)


           
    def load_params(self, load_path):
        with open("{}_global_cnv2d.pkl".format(load_path), 'rb') as fid:
            self.global_cnv1d = pkl.load(fid)

    def set_similar_task(self, task_id):
        self.conv2d.weight = copy.deepcopy(self.global_cnv1d[task_id].weight)
        if self.conv2d.bias is not None:
            self.conv2d.bias = copy.deepcopy(self.global_cnv1d[task_id].bias)



######################################################################################
class ConvTranspose2d_SVD(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super(ConvTranspose2d_SVD, self).__init__()
        self.conv_t = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=(kernel_size, kernel_size),
                        stride=(stride, stride),
                        padding=(padding, padding), bias=bias)
        self.global_conv_t = {}  
        self.conv_t.reset_parameters()
            
    def forward(self, inp, task_id=-1, task_ord_list=[]):
        x = inp
        if task_id == -1:
            return self.conv_t(x)
        else:
            conv_t = self.global_conv_t[task_id]
            return conv_t(x)
        


    def update_global(self, task_id):
        self.global_conv_t[task_id] = copy.deepcopy(self.conv_t)
        self.conv_t.reset_parameters()

        
    def get_task_params(self, prefix, task_id):
        conv_t_params = []
        if task_id == -1:
            conv_t = self.conv_t
        else:
            conv_t = self.global_conv_t[task_id]
        
        for name, param  in conv_t.named_parameters():
            conv_t_params.append(("{}.{}.conv_t.{}".format(prefix, name, task_id), param))
        
        return conv_t_params
    

    def save_params(self, save_path):
        with open("{}_global_conv_t.pkl".format(save_path), 'wb') as fid:
            pkl.dump(self.global_conv_t, fid)


           
    def load_params(self, load_path):
        with open("{}_global_conv_t.pkl".format(load_path), 'rb') as fid:
            self.global_conv_t = pkl.load(fid)

    def set_similar_task(self, task_id):
        self.conv_t.weight = copy.deepcopy(self.global_conv_t[task_id].weight)
        if self.conv_t.bias is not None:
            self.conv_t.bias = copy.deepcopy(self.global_conv_t[task_id].bias)


######################################################################################

# Matrix Transformation
#
#
# 
#  ###############################################################################

class FCBlock_Mtx(nn.Module):
    def __init__(self, fin, fout, bias=False, num_heads = 4, \
                ker_sz = (3, 3), stride = 1,padding = 1, \
                mode = "avg", conv_channels = 1, dilation=1):
        super(FCBlock_Mtx, self).__init__()
        self.fc = nn.Linear(fin, fout, bias=bias)
        self.global_wts = {}
        self.global_b = {}
        self.bias = bias
        torch.nn.init.xavier_uniform_(self.fc.weight)
        if self.bias:
            self.fc.bias.data.fill_(0.01)

        self.Mtx_fc = Mtx_fc(fin, fout, bias=bias, num_heads = num_heads, \
                             ker_sz = ker_sz, stride = stride, \
                             padding = padding, mode = mode, conv_channels = conv_channels, dilation = dilation)
        

    def set_parent(self, val):
        self.Mtx_fc.set_parent(val)

    def set_grad(self, val=False):
        self.fc.weight.requires_grad = val
        if self.fc.bias is not None:
            self.fc.bias.requires_grad = val
        self.Mtx_fc.set_grad(not val)

  
    def set_ker_size(self, ker_size):
        self.Mtx_fc.set_ker_size(ker_size)
            
    def forward(self, inp, task_id=-1, task_ord_list=[]):
        b_fc = None
        w_fc = None
        if task_id == -1:
            if self.Mtx_fc.parent_task:
                W_fc = self.fc.weight
                b_fc = self.fc.bias
            else:
                parent_task_id = task_ord_list[0]
                task_ord_list = task_ord_list[1:]
                W_fc = self.global_wts[parent_task_id]
                b_fc = self.global_b[parent_task_id]
        else:
            if self.Mtx_fc.parent_task:
                W_fc = self.global_wts[task_id]
                b_fc = self.global_b[task_id]
            else:
                parent_task_id = task_ord_list[0]
                task_ord_list = task_ord_list[1:]
                W_fc = self.global_wts[parent_task_id]
                b_fc = self.global_b[parent_task_id]
        if self.Mtx_fc.parent_task:
            return F.linear(inp, W_fc, b_fc)
        else:     
            return self.Mtx_fc(inp, W_fc, b=b_fc, task_id=task_id, task_ord_list=task_ord_list)
    

    def update_global(self, task_id):
        self.global_wts[task_id] = my_copy(self.fc.weight)
        self.global_b[task_id] = my_copy(self.fc.bias)
        self.Mtx_fc.update_global(task_id)

    def disp_conv_list(self, save_path):
        self.Mtx_fc.disp_conv_list(save_path)
    def set_similar_task(self, task_id):
        self.Mtx_fc.set_similar_task(task_id)

    def get_extra_params_num(self,curr_task=1):
        num_params = 0
        if curr_task:
            num_params += self.fc.weight.numel()
            if self.fc.bias is not None:
                num_params += self.fc.bias.numel()
        else:
            for wt, b in zip(self.global_wts.values(), self.global_b.values()):
                num_params += wt.numel()
                if b is not None:
                    num_params += b.numel()
        
        num_params += self.Mtx_fc.get_extra_params_num(curr_task)
        return num_params
    
    
    def get_init_params_num(self):
        
        num_params = self.fc.weight.numel()
        if self.fc.bias is not None:
            num_params += self.fc.bias.numel()
        return num_params
    
    def save_params(self, save_path):
        param_dict = {'wts': self.global_wts, 'bias': self.global_b}
        with open("{}_global_fcblk_mtx.pkl".format(save_path), 'wb') as fid:
            pkl.dump(param_dict, fid)

        self.Mtx_fc.save_params(save_path)

    def load_params(self, load_path):
        with open("{}_global_fcblk_mtx.pkl".format(load_path), 'rb') as fid:
            param_dict = pkl.load(fid)
            self.global_wts = param_dict['wts']
            self.global_b = param_dict['bias']

        self.Mtx_fc.load_params(load_path)


class Mtx_fc(nn.Module):
    def __init__(self, fin, fout, bias=True, num_heads = 4, ker_sz = (4, 4), stride = 1, \
                 padding = 1, mode = "avg", conv_channels = 1, dilation = 1):
        super(Mtx_fc, self).__init__()
        self.stride = stride
        self.mode = mode
        self.num_heads = num_heads
        self.conv_channels = conv_channels
        self.ker_size = ker_sz
        self.padding = padding
        self.dilation = dilation



        self.conv_list = ModuleList()
        self.global_sk_wt = {}
        self.sk_wt = None
        # self.global_sk_wt2 = {}
        # self.sk_wt2 = None
        # for i in range(self.num_heads):
        #     self.conv_list.append(nn.Conv2d(1, self.conv_channels, self.ker_size, padding=self.padding, stride=self.stride, dilation=self.dilation).cuda())
        
        
        # self.sk_wt = nn.Parameter(torch.randn(4, 1, 1).cuda())
        self.fin = fin
        self.fout = fout
        self.global_conv_list = {}
        self.parent_task = True
        self.global_b = {}
        if bias:
            self.bias_gamma = nn.Parameter(torch.tensor(1.))
            self.bias_beta  = nn.Parameter(torch.tensor(0.))
            
        else:
            self.register_parameter('bias_gamma', None)
            self.register_parameter('bias_beta', None)
        self.reset_parameters()
        
    def set_parent(self, val=True):
        self.parent_task = val

    def set_ker_size(self, ker_size):
        
        del self.conv_list[:]
        self.ker_size = ker_size
        padding = int((self.ker_size - 1)/2)
        self.padding = (padding, padding)
        for i in range(self.num_heads):
            self.conv_list.append(nn.Conv2d(1, self.conv_channels, self.ker_size, padding=self.padding, stride=self.stride, dilation=self.dilation).cuda())


    def reset_parameters(self):
        if not self.parent_task:
            for i in range(self.num_heads):
                self.conv_list[i].reset_parameters()
        
        if self.sk_wt is not None:
            nn.init.ones_(self.sk_wt)
        if self.bias_gamma is not None:
            nn.init.ones_(self.bias_gamma)
            nn.init.ones_(self.bias_beta)
    
    def set_grad(self, val=False):
        self.kernel.requires_grad = val
        if self.bias_gamma is not None:
            self.bias_gamma.requires_grad = val
            self.bias_beta.requires_grad = val
    
    def get_pos(self, mat_sz, ker_sz, stride = 1):
        x_pos = ((mat_sz[0] - ker_sz[0])/stride)+1
        y_pos = ((mat_sz[1] - ker_sz[1])/stride)+1
        return (int(x_pos), int(y_pos))

    def tr_apply_kernel_tr(self, mat, sk_wt, conv_list):
        input_shape = mat.shape
        all_head_size = input_shape[0]
        assert(all_head_size%self.num_heads == 0)
        mat = mat.view((self.num_heads, int(all_head_size/self.num_heads), mat.shape[1]))
        # print(mat.shape)
        out = torch.zeros_like(mat)
        
        for i in range(mat.shape[0]):
            # out[i] = torch.max(conv_list[i](mat[i].unsqueeze(0).unsqueeze(1)).squeeze(), dim = 0)[0]
            try:
                out[i] = conv_list[i](mat[i].unsqueeze(0).unsqueeze(1)).squeeze()
            except:
                print(conv_list)
        nsk_wt = torch.sigmoid(sk_wt)
        out = out + nsk_wt * mat
        return out.view(*input_shape)
    
    def trans_matx(self, mat, sk_wt, conv_list, stride = 1, mode = "avg"):
        return self.tr_apply_kernel_tr(mat, sk_wt, conv_list)

    def forward(self, input, W, b=None, task_id=-1, task_ord_list=[]):

        W_i = None
        b_i = None
        # pass through previous tids in task order list
        if self.parent_task:
            W_i = W
            b_i = b
        else:
            W_0 = W
            b_0 = b

            if task_id > 0:
                W_i = self.trans_matx(W_0, self.global_sk_wt[task_id], \
                self.global_conv_list[task_id], self.stride, self.mode)



                if self.global_b[task_id][0] is not None:
                    b_gamma, b_beta = self.global_b[task_id]
                    b_i = b_0 *b_gamma + b_beta
            else:

                W_i = self.trans_matx(W_0, self.sk_wt, self.conv_list, self.stride, self.mode)
                if self.bias_gamma is not None:
                    b_i = b_0*self.bias_gamma + self.bias_beta
                

        return F.linear(input, W_i, b_i)
    
    def update_global(self, task_id):
        new_list = []
        if not self.parent_task:

            for i in range(self.num_heads):
                new_list.append(copy.deepcopy(self.conv_list[i]))
            self.global_conv_list[task_id] = new_list
            self.global_b[task_id] = (my_copy(self.bias_gamma), my_copy(self.bias_beta))
            self.global_sk_wt[task_id] = my_copy(self.sk_wt)
            
        else:
            for i in range(self.num_heads):
                self.conv_list.append(nn.Conv2d(1, self.conv_channels, self.ker_size, padding=self.padding, stride=self.stride, dilation=self.dilation).cuda())
            
            
            self.sk_wt = nn.Parameter(torch.randn(4, 1, 1).cuda())
                                    
        self.reset_parameters()


    def disp_conv_list(self, save_path):
        if self.parent_task:
            return
        os.makedirs(save_path, exist_ok=True)
        for i in range(self.num_heads):
            fname = "{}/conv_head_{}.png".format(save_path, i+1)
            filter = self.conv_list[i].weight.cpu().data.clone()
            visTensor(filter, fname, ch=0, allkernels=False)

    def set_similar_task(self, task_id):
        if task_id == 0:
            return

        
        for i in range(self.num_heads):
            self.conv_list[i].weight = my_copy(self.global_conv_list[task_id][i].weight)
            self.conv_list[i].bias = my_copy(self.global_conv_list[task_id][i].bias)
        
        if self.bias_gamma is not None:
            (self.bias_gamma, self.bias_beta) = my_copy(self.global_b[task_id])
        
        if self.sk_wt is not None:
            self.sk_wt = my_copy(self.global_sk_wt[task_id])
        return


            
    def get_extra_params_num(self, curr_task=1):
        num_params = 0
        if curr_task:
            num_params += self.kernel1.numel()
            if self.bias_gamma is not None:
                 num_params += self.bias_gamma.numel() + self.bias_beta.numel()
        if not curr_task:
            for u, b in zip(self.global_kernel.values(), self.global_b.values()):
                num_params += u.numel()
                if b[0] is not None:
                    b_gamma, b_beta = b[0], b[1]
                    num_params += b_gamma.numel() + b_beta.numel()
        
        return num_params


    def save_params(self, save_path):
        param_dict = {'conv_list': self.global_conv_list, 'bias': self.global_b, 'sk_wt': self.global_sk_wt}
        with open("{}_global_mtx_fc.pkl".format(save_path), 'wb') as fid:
            pkl.dump(param_dict, fid)



    def load_params(self, save_path):
        with open("{}_global_mtx_fc.pkl".format(save_path), 'rb') as fid:
            param_dict = pkl.load(fid)
            self.global_conv_list = param_dict['conv_list']
            self.global_b = param_dict['bias']
            self.global_sk_wt = param_dict['sk_wt']

            for i in range(self.num_heads):
                self.conv_list.append(nn.Conv2d(1, self.conv_channels, self.ker_size, padding=self.padding, stride=self.stride, dilation=self.dilation).cuda())
            
            
            self.sk_wt = nn.Parameter(torch.randn(4, 1, 1).cuda())

#########################################################################################################################
# ConvBlock(sharing filters for tokenizer)
class AdaFM_Conv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size = (3, 3), bias=True, stride=1, padding=1):
        super().__init__()

        self.global_num_task = 0
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.parent_task = True
        self.stride = 1
        self.padding = padding
        self.kernel_size = kernel_size
        self.conv_t = nn.Conv2d(self.in_channel, self.out_channel, \
                                self.kernel_size, stride=self.stride, \
                                padding=self.padding, bias=bias)
        self.global_wts = {}
        self.global_b = {}
        self.reset_parameters()

    def set_parent(self, val=True):
        self.parent_task = val

    def reset_parameters(self):
        self.conv_t.reset_parameters()

    def set_grad(self, val=False):
        self.conv_t.weight.requires_grad = val
        if self.conv_t.bias is not None:
            self.conv_t.bias.requires_grad = val

    def forward(self, input, W, b=None, task_id=-1,  task_ord_list=[]):    
        W_f = None
        b_f = None

        if task_id == -1: 
            W_f = torch.cat((W, self.conv_t.weight), dim=0)
            if b is not None:
                b_f = torch.cat((b, self.conv_t.bias), dim = 0)

        else: 
            W_t = self.global_wts[task_id]
            b_t = self.global_b[task_id]
            W_f = torch.cat((W, W_t), dim=0)
            if b is not None:
                b_f = torch.cat((b, b_t), dim = 0)

        return torch.nn.functional.conv2d(input, W_f, bias=b_f, stride=self.stride, padding=self.padding)

    def update_global(self, task_id):
        self.global_wts[task_id] = my_copy(self.conv_t.weight)           
        self.global_b[task_id] = my_copy(self.conv_t.bias)

        self.reset_parameters()

    def set_similar_task(self, task_id):
        self.conv_t.weight = my_copy(self.global_wts[task_id])
        self.conv_t.bias = my_copy(self.global_b[task_id])
        

    def get_extra_params_num(self, curr_task=1):
        num_params = 0
        if curr_task:
            num_params += self.style_gamma.numel() + self.style_beta.numel()
            if self.bias is not None:
                 num_params += self.b.numel()
        if not curr_task:
            for gamma, beta, b in zip(self.global_gamma.values(), \
                                      self.global_beta.values(), self.global_b.values()):
                num_params += gamma.numel() + beta.numel()
                if b is not None:
                    num_params += b.numel()
        
        return num_params

    def save_params(self, save_path):
        param_dict = {'wts': self.global_wts, 'bias': self.global_b}
        with open("{}_global_adafm_conv.pkl".format(save_path), 'wb') as fid:
            pkl.dump(param_dict, fid)


    def load_params(self, save_path):
        with open("{}_global_adafm_conv.pkl".format(save_path), 'rb') as fid:
            param_dict = pkl.load(fid)
            self.global_wts = param_dict['wts']
            self.global_b = param_dict['bias']

class Conv_SVD(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, bias=True, split_ratio = 0.8):
        super().__init__()
        # Attributes
        self.bias = bias
        self.global_wts = {}
        self.global_b = {}
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.split_ratio = split_ratio
        self.shared_channels = int(math.floor(split_ratio*self.out_channels))

        # Submodules
        self.conv = nn.Conv2d(self.in_channels, self.shared_channels, self.kernel_size, \
                              stride=self.stride, padding=self.padding, bias=bias)
        self.AdaFM_Conv = AdaFM_Conv(self.in_channels, self.out_channels - self.shared_channels, \
                                    kernel_size = self.kernel_size, bias=bias, \
                                    stride=self.stride, padding=self.padding)
        self.set_grad(True)

    def set_grad(self, val=False):
        self.conv.weight.requires_grad = val
        if self.conv.bias is not None:
            self.conv.bias.requires_grad = val
        self.AdaFM_Conv.set_grad(True)

    def set_parent(self, val):
        self.AdaFM_Conv.set_parent(val)
        self.set_grad(val)

    def forward(self, x, task_id=-1, task_ord_list=[]):
        if task_id == -1:
            if self.AdaFM_Conv.parent_task:
                W = self.conv.weight
                b = self.conv.bias
            else:
                parent_task_id = task_ord_list[0]
                task_ord_list = task_ord_list[1:]
                W = self.global_wts[parent_task_id]
                b = self.global_b[parent_task_id]
        else:
            if self.AdaFM_Conv.parent_task:
                W = self.global_wts[task_id]
                b = self.global_b[task_id]
            else:
                parent_task_id = task_ord_list[0]
                task_ord_list = task_ord_list[1:]
                W = self.global_wts[parent_task_id]
                b = self.global_b[parent_task_id]

        return self.AdaFM_Conv(x, W, b, task_id=task_id, task_ord_list=task_ord_list)

            
    def update_global(self, task_id):
        if self.AdaFM_Conv.parent_task:
            assert(task_id == 0)
            self.global_wts[task_id] = my_copy(self.conv.weight)
            self.global_b[task_id] = my_copy(self.conv.bias)
            self.AdaFM_Conv.update_global(task_id)
            self.set_grad()
        else:
            self.AdaFM_Conv.update_global(task_id)
                                   
    def set_similar_task(self, task_id):
        self.AdaFM_Conv.set_similar_task(task_id)



    def get_extra_params_num(self,curr_task=1):
        num_params = 0
        if curr_task:
            num_params += self.conv.weight.numel()
            if self.conv.bias is not None:
                num_params += self.conv.bias.numel()
        else:
            for wt, b in zip(self.global_wts.values(), self.global_b.values()):
                num_params += wt.numel()
                if b is not None:
                    num_params += b.numel()
        
        num_params += self.AdaFM_Conv.get_extra_params_num(curr_task)
        return num_params

    def get_init_params_num(self):
        
        num_params = self.conv.weight.numel()
        if self.conv.bias is not None:
            num_params += self.conv.bias.numel()

        return num_params
    
    def save_params(self, save_path):
        param_dict = {'wts': self.global_wts, 'bias': self.global_b}
        with open("{}_global_conv_svd.pkl".format(save_path), 'wb') as fid:
            pkl.dump(param_dict, fid)

        self.AdaFM_Conv.save_params(save_path)

    def load_params(self, save_path):
        with open("{}_global_conv_svd.pkl".format(save_path), 'rb') as fid:
            param_dict = pkl.load(fid)
            self.global_wts = param_dict['wts']
            self.global_b = param_dict['bias']
        self.AdaFM_Conv.load_params(save_path) 






