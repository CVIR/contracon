import torch
from torch.nn import Module, ModuleList, Linear, Dropout, LayerNorm, Identity, Parameter, init
import torch.nn.functional as F
from .stochastic_depth import DropPath
from .model_parts import *
import torch.nn as nn
import pickle as pkl
def round_to_nearest(input_size, width_mult, num_heads, min_value=1):
    new_width_mult = round(num_heads * width_mult)*1.0/num_heads
    input_size = int(new_width_mult * input_size)
    new_input_size = max(min_value, input_size)
    return new_input_size





class Attention(Module):
    """
    Obtained from timm: github.com:rwightman/pytorch-image-models
    """

    def __init__(self, dim, head_dim, num_heads=8, attention_dropout=0.1, projection_dropout=0.1, ker_sz=9, dilation=1, *args, **kwargs):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = self.head_dim ** -0.5
        self.attn_drop = Dropout(attention_dropout)
        self.dim = dim
        self.all_head_size = self.num_heads * self.head_dim
        self.ker_size = ker_sz
        self.stride = 1
        self.dilation = dilation
        self.eff_ker_size = self.dilation*(self.ker_size-1) + 1
        self.padding = int((self.eff_ker_size - 1)/2)
        self.eff_ker_size2 = self.dilation * (self.ker_size - 1) + 1
        self.padding2 = int((self.eff_ker_size2 - 1)/2)
        self.conv_channels = 1 
        self.query = FCBlock_Mtx(dim, self.all_head_size, bias=False, num_heads = num_heads, ker_sz=(self.eff_ker_size, self.eff_ker_size2), stride=self.stride,padding=(self.padding, self.padding2), conv_channels=self.conv_channels, dilation=self.dilation)
        self.key = FCBlock_Mtx(dim, self.all_head_size, bias=False, num_heads = num_heads, ker_sz=(self.eff_ker_size, self.eff_ker_size2), stride=self.stride,padding=(self.padding, self.padding2), conv_channels=self.conv_channels, dilation=self.dilation)
        self.value = FCBlock_Mtx(dim, self.all_head_size, bias=False, num_heads = num_heads, ker_sz=(self.eff_ker_size, self.eff_ker_size2), stride=self.stride,padding=(self.padding, self.padding2), conv_channels=self.conv_channels, dilation=self.dilation)
        # BertSelfOutput
        self.proj = Linear(self.all_head_size, dim, bias=False)
        self.proj_drop = Dropout(projection_dropout)
   



    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_heads, self.head_dim)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)
    
    def forward(self, x, task_id, task_ord_list):
        B, N, C = x.shape

        query = self.query(x, task_id, task_ord_list)
        key = self.key(x, task_id, task_ord_list)
        value = self.value(x, task_id, task_ord_list)

        
        q = self.transpose_for_scores(query)
        k = self.transpose_for_scores(key)
        v = self.transpose_for_scores(value)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        
        
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        


        x = (attn @ v).permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = x.size()[:-2] + (self.all_head_size,)
        x = x.view(*new_context_layer_shape)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    
    def update_global(self, task_id):
        self.query.update_global(task_id)
        self.key.update_global(task_id)
        self.value.update_global(task_id)

    def set_similar_task(self, task_id):
        self.query.set_similar_task(task_id)
        self.key.set_similar_task(task_id)
        self.value.set_similar_task(task_id)

    def set_ker_size(self, ker_size):
        self.query.set_ker_size(ker_size)
        self.key.set_ker_size(ker_size)
        self.value.set_ker_size(ker_size)

    def set_parent(self, val):
        self.query.set_parent(val)
        self.key.set_parent(val)
        self.value.set_parent(val)

    def set_grad(self, val):
        self.query.set_grad(val)
        self.key.set_grad(val)
        self.value.set_grad(val)

    def set_fhid(self, ratio):
        self.query.set_fhid(ratio)
        self.key.set_fhid(ratio)
        self.value.set_fhid(ratio)
    
    def get_extra_params_num(self,curr_task=1):
        num_params = 0
        num_params+= self.query.get_extra_params_num(curr_task)
        num_params+= self.key.get_extra_params_num(curr_task)
        num_params+= self.value.get_extra_params_num(curr_task)
        return num_params

    def disp_conv_list(self, save_path):
        self.query.disp_conv_list('{}_query'.format(save_path))
        self.key.disp_conv_list('{}_key'.format(save_path))
        self.value.disp_conv_list('{}_value'.format(save_path))
    
    def save_params(self, save_path):
        self.query.save_params('{}_query'.format(save_path))
        self.key.save_params('{}_key'.format(save_path))
        self.value.save_params('{}_value'.format(save_path))        

    def load_params(self, load_path):
        self.query.load_params('{}_query'.format(load_path))
        self.key.load_params('{}_key'.format(load_path))
        self.value.load_params('{}_value'.format(load_path))     
    
    def get_init_params_num(self):
        num_params = 0
        num_params+= self.query.get_init_params_num()
        num_params+= self.key.get_init_params_num()
        num_params+= self.value.get_init_params_num()
        num_params += sum(p.numel() for p in self.proj.parameters())
        return num_params

class TransformerEncoderLayer(Module):
    """
    Inspired by torch.nn.TransformerEncoderLayer and timm.
    """

    def __init__(self, d_model, head_dim, nhead, dim_feedforward=2048, dropout=0.1,
                 attention_dropout=0.1, drop_path_rate=0.1, seq_len=256, ker_sz=9, dilation=1):
        super(TransformerEncoderLayer, self).__init__()
        self.d_model = d_model
        self.pre_norm = LayerNorm(self.d_model)
        self.global_pre_norm = {}
        self.dim_feedforward = dim_feedforward
        self.self_attn = Attention(dim=self.d_model, head_dim=head_dim, num_heads=nhead,
                                   attention_dropout=attention_dropout, projection_dropout=dropout, ker_sz=ker_sz, dilation=dilation, seq_len=seq_len)

        self.linear1 = Linear(self.d_model, self.dim_feedforward)
        self.dropout1 = Dropout(dropout)
        self.norm1 = LayerNorm(self.d_model)
        self.global_norm1 = {}
        self.linear2 = Linear(self.dim_feedforward, self.d_model)

        self.dropout2 = Dropout(dropout)

        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0 else Identity()
        self.parent_task = True
        self.activation = F.gelu
        self.reset_parameters()
        
    def reset_parameters(self):
        self.pre_norm.reset_parameters()
        self.norm1.reset_parameters()

    def forward(self, src, task_id=-1, task_ord_list=[]):
        if task_id == -1:
            pre_norm = self.pre_norm
            norm1 = self.norm1
        else:
            pre_norm = self.global_pre_norm[task_id]
            norm1 = self.global_norm1[task_id]
        src = src + self.drop_path(self.self_attn(pre_norm(src), task_id=task_id, task_ord_list=task_ord_list))
        src = norm1(src)

        src2 = self.linear2(self.dropout1(self.activation(self.linear1(src))))
        src = src + self.drop_path(self.dropout2(src2))
        return src


    def modify_params(self, kernel_size=2):
        self.self_attn.modify_params(kernel_size)

    def set_parent(self, val):
        self.self_attn.set_parent(val)
        self.parent_task = val


    def update_global(self, task_id):
        self.global_pre_norm[task_id] = my_copy(self.pre_norm)
        self.global_norm1[task_id] = my_copy(self.norm1)
        self.reset_parameters()
        self.self_attn.update_global(task_id)

    def set_similar_task(self, task_id):
        self.self_attn.set_similar_task(task_id)
        self.pre_norm = copy.deepcopy(self.global_pre_norm[task_id])
        self.norm1 = copy.deepcopy(self.global_norm1[task_id])

    def set_ker_size(self, ker_size):
        self.self_attn.set_ker_size(ker_size)        
    def get_task_params(self, prefix, task_id):
        if task_id == -1:
            resweight = self.resweight
        else:
            resweight = self.global_resweights[task_id]
        named_param = {'{}.resweight.{}'.format(prefix, task_id): resweight}
        total_params = self.self_attn.get_task_params("{}.self_attn".format(prefix), task_id) + list(named_param.items())
        return total_params

    def disp_conv_list(self, save_path):
        self.self_attn.disp_conv_list(save_path)    
    def save_params(self, save_path):
        param_dict = {'pre_norm': self.global_pre_norm, 'norm1': self.global_norm1}
        with open("{}_global_enc_norms.pkl".format(save_path), 'wb') as fid:
            pkl.dump(param_dict, fid)
        self.self_attn.save_params("{}_self_attn".format(save_path))

           
    def load_params(self, load_path):
        param_dict = {'pre_norm': self.global_pre_norm, 'norm1': self.global_norm1}
        with open("{}_global_enc_norms.pkl".format(load_path), 'rb') as fid:
            param_dict = pkl.load(fid)
            self.global_pre_norm = param_dict['pre_norm']
            self.global_norm1 = param_dict['norm1']
        self.self_attn.load_params("{}_self_attn".format(load_path))
            

    
    
class TransformerClassifier(Module):
    def __init__(self,
                 seq_pool=True,
                 embedding_dim=768,
                 num_layers=12,
                 head_dim=64,
                 num_heads=12,
                 mlp_ratio=4.0,
                 initial_num_classes=1000,
                 num_classes=1000,
                 dropout=0.1,
                 attention_dropout=0.1,
                 stochastic_depth=0.1,
                 positional_embedding='sine',
                 sequence_length=None,
                 *args, **kwargs):
        super().__init__()

        positional_embedding = positional_embedding if \
            positional_embedding in ['sine', 'learnable', 'none'] else 'sine'
        dim_feedforward = int(embedding_dim * mlp_ratio)
        
        self.embedding_dim = embedding_dim
        self.sequence_length = sequence_length
        self.seq_pool = seq_pool
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.parent_task = True
        assert sequence_length is not None or positional_embedding == 'none', \
            f"Positional embedding is set to {positional_embedding} and" \
            f" the sequence length was not specified."

        if not seq_pool:
            sequence_length += 1
            self.class_emb = Parameter(torch.zeros(1, 1, self.embedding_dim),
                                       requires_grad=True)
        else:
            self.attention_pool_SVD = FCBlock_SVD(self.embedding_dim, 1)
            # print(self.embedding_dim * 3)

        if positional_embedding != 'none':
            if positional_embedding == 'learnable':
                self.positional_emb = Parameter(torch.zeros(1, sequence_length, embedding_dim),
                                                requires_grad=True)
                init.trunc_normal_(self.positional_emb, std=0.2)
            else:
                self.positional_emb = Parameter(self.sinusoidal_embedding(sequence_length, embedding_dim),
                                                requires_grad=False)
        else:
            self.positional_emb = None

        self.dropout = Dropout(p=dropout)
        dpr = [x.item() for x in torch.linspace(0, stochastic_depth, num_layers)]
        self.blocks = ModuleList([
            TransformerEncoderLayer(d_model=embedding_dim, head_dim=head_dim, nhead=num_heads,
                                    dim_feedforward=dim_feedforward, dropout=dropout,
                                    attention_dropout=attention_dropout, drop_path_rate=dpr[i], seq_len=self.sequence_length, ker_sz = kwargs['ker_sz'], dilation = kwargs['dilation'])
            for i in range(num_layers)])

        self.fc_SVD = FCBlock_SVD(embedding_dim, num_classes, initial_num_classes)
        self.apply(self.init_weight)


    def set_parent(self, val):
        self.parent_task = val
        for blk in self.blocks:
            blk.set_parent(val)

            
    def update_global(self, task_id):

        for blk in self.blocks:
            blk.update_global(task_id)        
        if self.seq_pool:
            self.attention_pool_SVD.update_global(task_id)
        self.fc_SVD.update_global(task_id)

    def disp_conv_list(self, save_path):
        for i, blk in enumerate(self.blocks):
            path = '{}/{}'.format(save_path, i)
            blk.disp_conv_list(path)

    def set_similar_task(self, task_id):
        for blk in self.blocks:
            blk.set_similar_task(task_id)
        if self.seq_pool:
            self.attention_pool_SVD.set_similar_task(task_id)
        self.fc_SVD.set_similar_task(task_id)        
    def set_ker_size(self, ker_size):
        for blk in self.blocks:
            blk.set_ker_size(ker_size)       

    def forward(self, x, task_id=-1, task_ord_list=[]):

        if self.positional_emb is None and x.size(1) < self.sequence_length:
            x = F.pad(x, (0, 0, 0, self.n_channels - x.size(1)), mode='constant', value=0)

        if not self.seq_pool:
            cls_token = self.class_emb.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_token, x), dim=1)

        if self.positional_emb is not None:
            x += self.positional_emb

        x = self.dropout(x)
        out_x_list = []
        avg_x = 0.
        for i, blk in enumerate(self.blocks):
            x = blk(x, task_id=task_id, task_ord_list = task_ord_list)
           
            if i > 2:
                 out_x_list.append(x)
        avg_x = torch.cat(out_x_list, 2)
        if self.seq_pool:
            x = torch.matmul(F.softmax(self.attention_pool_SVD(x, \
                task_id=task_id, task_ord_list=task_ord_list), dim=1).transpose(-1, -2), x).squeeze(-2)
        else:
            x = x[:, 0]
        out_x = self.fc_SVD(x, task_id=task_id, task_ord_list=task_ord_list)
        return out_x, avg_x



    
    def update_embeddings(self, x, y):
        pass

        
    def get_task_params(self, prefix, task_id):
        total_task_params = []
        for ix, blk in enumerate(self.blocks):
            total_task_params += blk.get_task_params("{}.blocks.{}".format(prefix, ix), task_id)
        if self.seq_pool:
            total_task_params +=  self.attention_pool_SVD.get_task_params("{}.attention_pool_SVD".format(prefix), task_id=task_id)

        total_task_params += self.fc_SVD.get_task_params("{}.fc_SVD".format(prefix), task_id=task_id)        
        return total_task_params

    def modify_params(self, kernel_size=2):
        for blk in self.blocks:
            blk.modify_params(kernel_size)
            

    def save_params(self, save_path):
        for i, blk in enumerate(self.blocks):
            blk.save_params("{}_{}".format(save_path, i))
        if self.seq_pool:
            self.attention_pool_SVD.save_params('{}_attn_pool'.format(save_path))
        self.fc_SVD.save_params('{}_out_fc'.format(save_path))
            
    def load_params(self, save_path):
        for i, blk in enumerate(self.blocks):
            blk.load_params("{}_{}".format(save_path, i))
        if self.seq_pool:
            self.attention_pool_SVD.load_params('{}_attn_pool'.format(save_path))
        self.fc_SVD.load_params('{}_out_fc'.format(save_path)) 

    @staticmethod
    def init_weight(m):
        if isinstance(m, Linear):
            init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, Linear) and m.bias is not None:
                init.constant_(m.bias, 0)
        elif isinstance(m, LayerNorm):
            init.constant_(m.bias, 0)
            init.constant_(m.weight, 1.0)

    @staticmethod
    def sinusoidal_embedding(n_channels, dim):
        pe = torch.FloatTensor([[p / (10000 ** (2 * (i // 2) / dim)) for i in range(dim)]
                                for p in range(n_channels)])
        pe[:, 0::2] = torch.sin(pe[:, 0::2])
        pe[:, 1::2] = torch.cos(pe[:, 1::2])
        return pe.unsqueeze(0)


