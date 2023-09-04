import torch.nn as nn
from .utils.transformers import TransformerClassifier
from .utils.tokenizer import Tokenizer
import os
import torch

__all__ = ['cct_2', 'cct_4', 'cct_6', 'cct_7', 'cct_8',
           'cct_14', 'cct_16']


class CCT(nn.Module):
    def __init__(self,
                 img_size=224,
                 embedding_dim=768,
                 n_input_channels=3,
                 n_conv_layers=1,
                 kernel_size=7,
                 stride=2,
                 padding=3,
                 pooling_kernel_size=3,
                 pooling_stride=2,
                 pooling_padding=1,
                 *args, **kwargs):
        super(CCT, self).__init__()

        self.tokenizer = Tokenizer(n_input_channels=n_input_channels,
                                   n_output_channels=embedding_dim,
                                   kernel_size=kernel_size,
                                   stride=stride,
                                   padding=padding,
                                   pooling_kernel_size=pooling_kernel_size,
                                   pooling_stride=pooling_stride,
                                   pooling_padding=pooling_padding,
                                   max_pool=True,
                                   activation=nn.ReLU,
                                   n_conv_layers=n_conv_layers,
                                   conv_bias=False,
                                   split_ratio = kwargs['split_ratio'])

        self.classifier = TransformerClassifier(
            sequence_length=self.tokenizer.sequence_length(n_channels=n_input_channels,
                                                           height=img_size,
                                                           width=img_size),
            embedding_dim=embedding_dim,
            seq_pool=True,
            dropout=0.,
            attention_dropout=0.1,
            stochastic_depth=0.1,
            *args, **kwargs)


    def set_parent(self, val):
        self.tokenizer.set_parent(val)
        self.classifier.set_parent(val)
        
    def update_global(self, task_id):

        self.tokenizer.update_global(task_id)
        self.classifier.update_global(task_id)

        
    def forward(self, x, task_id=-1, task_ord_list=[]):
        x = self.tokenizer(x, task_id, task_ord_list = task_ord_list)

        out_x, feat = self.classifier(x, task_id, task_ord_list = task_ord_list)
        return out_x, feat

    def update_embeddings(self, x, y):
        x = self.tokenizer(x, -1, [0])
        self.classifier.update_embeddings(x, y)
        
    def disp_conv_list(self, save_path):
        self.classifier.disp_conv_list(save_path)
    def set_similar_task(self, task_id):
        self.tokenizer.set_similar_task(task_id)
        self.classifier.set_similar_task(task_id)


    def set_ker_size(self, ker_size):
        self.classifier.set_ker_size(ker_size)

    def get_task_params(self, task_id):
        return self.classifier.get_task_params("classifier", task_id)

    def modify_params(self, kernel_size=2):
        self.classifier.modify_params(kernel_size)
    
    def save_params(self, save_path):
        os.makedirs(save_path, exist_ok = True)
        os.makedirs(f'{save_path}/state_dict', exist_ok = True)
        self.classifier.save_params("{}/classifier".format(save_path))
        self.tokenizer.save_params('{}/tokenizer'.format(save_path))

    def load_params(self, load_path):
        self.classifier.load_params("{}/classifier".format(load_path))
        self.tokenizer.load_params("{}/tokenizer".format(load_path))

    def load_base(self, load_path):
        self.tokenizer.load_state_dict(torch.load('{}/tokenizer_base.pth'.format(load_path)))
        self.classifier.load_state_dict(torch.load('{}/classifier_base.pth'.format(load_path)))
    
    def get_task(self):
        return len(self.classifier.fc_SVD.global_fcblk)


    def load_my_state_dict(self, state_dict):
 
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name not in own_state:
                 continue
            if isinstance(param, torch.nn.Parameter):
                # backwards compatibility for serialized parameters
                param = param.data
            try:
                own_state[name].copy_(param)
            except:
                pass


def _cct(num_layers, num_heads, mlp_ratio, embedding_dim,
         kernel_size=3, stride=None, padding=None,
         *args, **kwargs):
    stride = stride if stride is not None else max(1, (kernel_size // 2) - 1)
    padding = padding if padding is not None else max(1, (kernel_size // 2))
    return CCT(num_layers=num_layers,
               num_heads=num_heads,
               mlp_ratio=mlp_ratio,
               embedding_dim=embedding_dim,
               kernel_size=kernel_size,
               stride=stride,
               padding=padding,
               *args, **kwargs)





def cct_2(*args, **kwargs):
    return _cct(num_layers=2, num_heads=2, mlp_ratio=1, embedding_dim=128,
                *args, **kwargs)


def cct_4(*args, **kwargs):
    return _cct(num_layers=4, num_heads=2, mlp_ratio=1, embedding_dim=128,
                *args, **kwargs)


def cct_6(*args, **kwargs):
    return _cct(num_layers=6, num_heads=4, mlp_ratio=2, embedding_dim=256,
                *args, **kwargs)


def cct_7(*args, **kwargs):
    return _cct(num_layers=7, num_heads=4, mlp_ratio=2, embedding_dim=256,
                *args, **kwargs)

def cct_6_2(*args, **kwargs):
    return _cct(num_layers=6, num_heads=8, mlp_ratio=2, embedding_dim=256,
                *args, **kwargs)
def cct_8(*args, **kwargs):
    return _cct(num_layers=8, num_heads=4, mlp_ratio=2, embedding_dim=256,
                *args, **kwargs)


def cct_14(*args, **kwargs):
    return _cct(num_layers=10, num_heads=6, mlp_ratio=3, embedding_dim=384,
                *args, **kwargs)


def cct_16(*args, **kwargs):
    return _cct(num_layers=16, num_heads=6, mlp_ratio=3, embedding_dim=384,
                *args, **kwargs)


