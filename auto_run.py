import os
import torch
import torch.nn as nn
import numpy as np
import argparse
import copy
import math
import pdb
import torch.nn.functional as F
import re, random, collections
import pickle
from utils.losses import SoftTargetCrossEntropy, LabelSmoothingCrossEntropy
from timm.data.mixup import Mixup
import copy
import time


import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import argparse
from torch.optim.lr_scheduler import MultiStepLR, CosineAnnealingLR, CosineAnnealingWarmRestarts
from five_datasets.dataloader import five_datasets as five_loader
from five_datasets.dataloader.five_datasets import combine_x_y, get_five_loaders

import incremental_dataloader as datal
import src as models
from mixup import Mixup
from src.utils.augmentations import CIFAR10Policy




###########################################################################################################################################

parser = argparse.ArgumentParser()

parser.add_argument('--ker_sz', type=int, default=15, help='kernel_size of conv_mtx')
parser.add_argument('--num_tasks', type=int, default=10, help='number of tasks')
parser.add_argument('--num_classes', type=int, default=100, help='number of total classes')
parser.add_argument('--dil', type=int, default=1, help='dilation of conv_mtx')
parser.add_argument('--sr', type=float, default=1.0, help='split_ratio for tokenizer')
parser.add_argument('--nepochs', type=int, default=250, help='num of epochs')
parser.add_argument('--gpu_no', type=int, default=0, help='cuda gpu device number')
parser.add_argument('--is_task0', default=False, action='store_true', help='use this for training and saving task0 only')
parser.add_argument('--use_saved', default=False, action='store_true', help='use the saved model')
parser.add_argument('--dataset', choices=['cifar100', '5d',  'tin', 'imagenet100'], default='imagenet100', help="which dataset")
parser.add_argument('--data_path', type=str, default="./Datasets/tiny-imagenet-200/", help='path to dataset')
parser.add_argument('--scenario', choices=['til', 'cil'], default='til', help="which scenario [til | cil]")

model_args = parser.parse_args()

print(model_args)
is_task0 = model_args.is_task0


task0_model_path = 'task0_models/{}.pkl'.format(model_args.dataset)


test_transform = None
class args():
    data_path = model_args.data_path

    if model_args.dataset == 'cifar100':
        dataset = "cifar100"
        train_batch = 256
        test_batch = 256
        batch_size= 256
        inp_size = 32
        n_conv_layers = 1
        tok_kernel_size = 3
        num_classes = 100
        cct_val = 'cct_6'
        
    elif model_args.dataset == 'tin':
        dataset = "tinyimagenet"
        train_batch = 256
        test_batch = 256
        batch_size= 256
        inp_size = 64
        n_conv_layers = 2
        tok_kernel_size = 5
        num_classes = 200
        cct_val = 'cct_6'

    elif model_args.dataset == '5d':
        dataset = '5d'                                                                                                                                                                  
        train_batch = 256                                                                                                                                                               
        test_batch = 256                                                                                                                                                                
        batch_size= 256                                                                                                                                                                 
        inp_size = 32                                                                                                                                                                   
        n_conv_layers = 2                                                                                                                                                               
        tok_kernel_size = 3
        num_classes = 10                                                                                                                                                             
        cct_val = 'cct_7'    

    elif model_args.dataset == 'imagenet100':
        dataset = 'imagenet100'
        train_batch = 256
        test_batch = 256
        batch_size= 256
        inp_size = 224
        n_conv_layers = 3
        tok_kernel_size = 3
        num_classes = 100
        cct_val = 'cct_6'
    else:
        raise("Wrong dataset argument")
    num_task = model_args.num_tasks
    class_per_task = int(num_classes/model_args.num_tasks)
    print("CLASS_PER_TASK = ",class_per_task)
    workers = 4
    random_classes = False
    validation = 0
    overflow=False
    lr=0.01
    resume=False
    total_epoch=model_args.nepochs

    

if model_args.dataset == 'cifar100':
    test_transform = transforms.Compose([transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))])
elif model_args.dataset == 'tin':
    test_transform = transforms.Compose([transforms.Normalize((0.480, 0.448, 0.397), (0.277, 0.270, 0.282))])
elif model_args.dataset == 'imagenet100':
    test_transform = transforms.Compose([transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

    test_transform = transforms.Compose([transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])                                                                                                             
elif model_args.dataset == '5d':                                                                                                                                                                                          
    test_transform = [transforms.Compose([transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616])]),                                                                                              
                      transforms.Compose([transforms.Normalize((0.1,0.1, 0.1), (0.2752, 0.2752, 0.2752))]),                                                                                                      
                      transforms.Compose([transforms.Normalize([0.4377,0.4438,0.4728], [0.198,0.201,0.197])]),                                                                                                
                      transforms.Compose([transforms.Normalize((0.2190, 0.2190, 0.2190), (0.3318,0.3318, 0.3318))]),                                                                        
                      transforms.Compose([transforms.Normalize((0.4254, 0.4254, 0.4254), (0.4501, 0.4501, 0.4501))])                                                                                                      
                     ]                                                                                                                                                                                                    

  
else:
    pass


class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import argparse
from torch.optim.lr_scheduler import MultiStepLR, CosineAnnealingLR, CosineAnnealingWarmRestarts



def save_model(task,acc,model):
    print('Saving..')
    statem = {
        'net': model.state_dict(),
        'acc': acc,
    }
    fname=args.model_path
    if not os.path.isdir(fname):
        os.makedirs(fname)
    torch.save(statem, fname+'/ckpt_task'+str(task)+'.pth')

        
def load_model(task,model):
    fname=args.model_path
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    print(fname+'/ckpt_task'+str(task)+'.pth')
    checkpoint = torch.load(fname+'/ckpt_task'+str(task)+'.pth')
    model.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    return best_acc



def train(train_loader,epochs,task, model, task_ord_list, mixup_fn, test_loader, split_ratio):
    if task == 0 and args.dataset == 'imagenet100':
        model = nn.DataParallel(model)
    model.train()
    model.zero_grad()
    print("TASK ID = ", task)
    best_acc = 0
    best_model = None
    criterion = LabelSmoothingCrossEntropy().cuda()
    criterion2 = nn.MSELoss().cuda()
    params_to_update = []
    task_optim_list = []
    curr_params_to_update  = []

    for name, param in model.named_parameters():
        # print(name)
        if task:
            # pass


            if name.find('untok') >= 0:
                # print("OK")
                param.requires_grad = True
            elif name.find('_scale') > 0:
                param.requires_grad = False
            elif name.find('_shift') > 0:
                param.requires_grad = False
            elif name.find('Mtx_fc') > 0:
                param.requires_grad = True
            elif name.find('_SVD') > 0:
                param.requires_grad = True
            elif name.find('norm1') > 0:
                param.requires_grad = True
            elif name.find('pre_norm') > 0:
                param.requires_grad = True
            elif name.find('conv_layers') > 0:
                if split_ratio == 1.0:
                    param.requires_grad  = False
                else:
                    continue
            elif name.find('positional_emb') > 0:
                continue
            else:
                param.requires_grad  = False
        else:
            if name.find('untok') >=0:
                print("OK ", name)
                param.requires_grad = True
            # else:
            #     param.requires_grad = False
            elif name.find('_scale') > 0:
                param.requires_grad = False
            elif name.find('_shift') > 0:
                param.requires_grad = False
            elif name.find('Mtx_fc') > 0:
                param.requires_grad = False
            elif name.find('_SVD') > 0:
                param.requires_grad = True
            elif name.find('norm1') > 0:
                param.requires_grad = True
            elif name.find('pre_norm') > 0:
                param.requires_grad = True
            elif name.find('conv_layers') > 0:
                if split_ratio == 1.0:
                    param.requires_grad  = True
                else:
                    continue
            elif name.find('positional_emb') > 0:
                continue
            else:
                param.requires_grad  = True


    num_params = 0
    untok_params_to_update = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            num_params += param.numel()
            if name.find('fc_SVD') >=0:
                untok_params_to_update.append(param)
            else:
                params_to_update.append(param)

    print("No of trainable parameters for task {} is {}".format(task, num_params))

    meta_params = []

    optim = torch.optim.AdamW(params_to_update, lr=8e-4, weight_decay=6e-7)
    optim2 = torch.optim.SGD(untok_params_to_update, lr=5e-3)
    scheduler = CosineAnnealingWarmRestarts(optim, eta_min=1e-5, T_0=epochs//2)
    print(criterion)
    for epoch in range(epochs):
        since = time.time()
        model.train()
        train_loss = 0
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            if args.dataset != '5d':
                targets = targets - task * args.class_per_task
            else:
                if task == 0 or task == 2:
                    inputs = inputs[0]
            inputs, targets = inputs.cuda(), targets.cuda()
            optim.zero_grad()
            optim2.zero_grad()      
            outputs, curr_proto = model(inputs, task_id=-1, task_ord_list = task_ord_list)
            curr_proto = curr_proto.view(curr_proto.size(0), -1)
            loss = criterion(outputs, targets)
            loss.backward()
            optim.step()
            optim2.step()
            train_loss += loss.item()
        # print(train_loss)
        time_elapsed = time.time() - since
        since_t = time.time()
        if epoch > 300:
            acc = test(test_loader,task,model, task_ord_list)

        else:
            acc = test(test_loader,task,model, task_ord_list)
        time_elapsed_t = time.time() - since_t
        if acc > best_acc:
            best_acc = acc
            
            if task == 0:
                torch.save(model.state_dict(), task0_model_path)
            else:
                best_model = copy.deepcopy(model)
        print("[Train: ], [%d/%d: ], [Accuracy: %f], [Loss: %f] [Lr: %f]  --- [Training time: %f] [Testing time: %f]" 
            %(epoch,args.total_epoch,acc, train_loss/batch_idx,
            optim.param_groups[0]['lr'], time_elapsed, time_elapsed_t))

        scheduler.step()

    model.zero_grad()
    return best_model



def get_aug_img(task, inp):
    global policy
    aug_img =[]
    if args.dataset == '5d':
        if task == 0 or task == 2:
            new_inp = test_transform[task](inp).cuda()
        else:
            new_inp = inp.cuda()
    else:
        new_inp = test_transform(inp).cuda()
    for _ in range(6):
        aug_img.append(new_inp)
    
    if args.dataset == '5d' and (task != 0  and task != 2):
        aug = transforms.Compose([AddGaussianNoise(0., 1.)])
    else:
        aug = transforms.Compose([transforms.ToPILImage(),
                        CIFAR10Policy(),                        
                        transforms.ToTensor(),                        
                        ])
    for i in range(10):

        if args.dataset != '5d':
            tinp = test_transform(aug(inp)).cuda()
        else:
            if task == 0 or task == 2:
                tinp = test_transform[task](aug(inp)).cuda()
            else:
                tinp = aug(inp).cuda()
        aug_img.append(tinp)
    aug_img = torch.stack(aug_img, dim=0)
    return aug_img
def get_mean_output(task, inp2, task_id, model, task_ord_list):
    samples = torch.zeros((inp2.size(0), args.class_per_task)).cuda()
    for i in range(inp2.size(0)):
        rand_img = get_aug_img(task, inp2[i])
        with torch.no_grad():

            outputs, out_x = model(rand_img, task_id, task_ord_list)

            outputs = outputs.mean(dim=0)

        samples[i] = outputs
    return samples


def test(test_loader,task,model, task_ord_list, task_id=-1):

    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    cl_loss=0
    tcorrect=0
    gaussian_trfm = transforms.Compose([AddGaussianNoise(0., 1.)])
                
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            if args.dataset != '5d':
                targets = targets - task * args.class_per_task
            
            
            if args.dataset != '5d':
                inputs1 = test_transform(inputs).cuda()
            elif task == 0 or task == 2:
                inputs1 = test_transform[task](inputs).cuda()
            else:
                inputs1 = inputs.cuda()
            total += targets.size(0)
            inputs, targets = inputs.cuda(), targets.cuda()

            if inputs.shape[0]!=0:
                outputs, _ = model(inputs1, task_id=task_id, task_ord_list = task_ord_list)
                loss = criterion2(outputs, targets)


                test_loss += loss.item()
                _, predicted = outputs.max(1)
                correct += predicted.eq(targets).sum().item()


        
        acc = 100.*correct/total

        taskC = 1.0
        print("[Test Accuracy: %f], [Correct: %f]" %(acc,taskC))
    
    
  
    return acc




def check_task(task, inputs4, targets1, model,task_parent_dict, total_task=None):
    joint_entropy=[]
    
    batch_size = inputs4.size(0)
    if total_task is None:
        tot_task = task + 1
        if args.dataset != '5d':
            inputs4 = test_transform(inputs4)
        elif task == 0 or task == 2:
            inputs4 = test_transform[task](inputs4)
    else:
        tot_task = total_task
    with torch.no_grad():
        for task_id in range(tot_task):
            task_ord_list = get_task_ord_list(task_parent_dict, task_id)
            if len(task_ord_list)==0:
                model.set_parent(True)
            else:
                model.set_parent(False)
            model.eval()
            if total_task is None:
                if task_id == task:
                    task_id = -1

            
            if total_task is None:
                outputs, _ =model(inputs4, task_id, task_ord_list)
            else:
                outputs = get_mean_output(task, inputs4, task_id, model, task_ord_list)
                inputs5 = test_transform(inputs4).cuda()
                outputs2, _ =model(inputs5, task_id, task_ord_list)
                outputs2=F.softmax(outputs2,1)

            sout = F.softmax(outputs, 1)


            dist=-torch.sum(sout*torch.log(sout+0.0001),1)
            joint_entropy.append(dist)
        all_entropy=torch.zeros([inputs4.shape[0], tot_task]).cuda()
        for i in range(tot_task):
            all_entropy[:, i] = joint_entropy[i]
    ctask=torch.argmin(all_entropy, axis=1)==task
    correct=sum(ctask)
    
    return ctask, correct,all_entropy


def test_all(test_loader_list, model, task_parent_dict):

    global best_acc
    model.eval()
    acc_list = []
    taskC_list = []
    criterion = nn.CrossEntropyLoss().cuda()
    tot_task = len(test_loader_list)
    for task, test_loader in enumerate(test_loader_list):
        test_loss = 0
        correct = 0
        total = 0
        
        cl_loss=0
        tcorrect=0
        task_ord_list = [0]
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(test_loader):
                
                if args.dataset != '5d':
                    targets=targets-task*args.class_per_task
                targets = targets.cuda()
                if args.dataset != '5d':
                   inputs1 = test_transform(inputs).cuda()
                elif task == 0 or task == 2:
                    inputs1 = test_transform[task](inputs).cuda()
                else:
                    inputs1 = inputs.cuda()
                total += targets.size(0)
                
                if model_args.scenario == 'cil' and tot_task>0:
                    correct_sample,Ncorrect,_=check_task(task, inputs, targets, model,task_parent_dict, tot_task)
                    tcorrect+=Ncorrect
                    inputs1=inputs1[correct_sample]
                    targets=targets[correct_sample]
                if inputs.shape[0]!=0:
                    task_ord_list = get_task_ord_list(task_parent_dict, task)
                    if len(task_ord_list)==0:
                        model.set_parent(True)
                    else:
                        model.set_parent(False)
                    outputs,_ = model(inputs1, task, task_ord_list)

                    _, predicted = outputs.max(1)
                    correct += predicted.eq(targets).sum().item()
                
            if model_args.scenario == 'cil' and tot_task>0:
                taskC= tcorrect.item()/total
            else:
                taskC=1.0 
        acc = 100.*correct/total
        acc_list.append(acc)
        taskC_list.append(taskC)
        print("[Test Accuracy: %f], [Loss: %f] [Correct: %f]" %(acc,test_loss/batch_idx,taskC))
        
    

    print("AVERAGE ACCURACY = ", sum(acc_list)/len(acc_list))
    print("AVERAGE TASK CORRECT = ", sum(taskC_list)/len(taskC_list))
    return acc, taskC

    
    
  



def calc_task_similarity(train_loader,task_id, modelm, task_parent_dict):
    best_accuracy = 0
    best_task = 0
    for tid in range(task_id-1,-1,-1):
        task_ord_list = get_task_ord_list(task_parent_dict, tid)
        if len(task_ord_list) == 0:
            modelm.set_parent(True)
        else:
            modelm.set_parent(False)
        
        acc = compute_importance(train_loader, task_id, modelm, task_ord_list, tid)
        if acc > best_accuracy:
            best_accuracy = acc
            best_task = tid
    return best_task, best_accuracy

def get_task_ord_list(task_parent_dict, task_id):
    task_ord_list = []
    tid = task_parent_dict[task_id]
    while tid != task_parent_dict[tid]:
        task_ord_list.append(tid)
        tid = task_parent_dict[tid]
    if tid != task_id:
        task_ord_list.append(tid)
    task_ord_list = task_ord_list[::-1]
    return task_ord_list

def set_parent_task(task_id, p_task_id):
    task_parent_dict[task_id] = p_task_id  

def remove_module_from_state_dict(model_path):
    model_dict = torch.load(model_path)
    new_model_dict = {}
    for k, v in model_dict.items():
        if 'module.' in k:
            k = k.replace('module.','')
            new_model_dict[k] = copy.deepcopy(v)
    return new_model_dict

def change_tokenizer_state(dict, old_sr, new_sr, out_channel, n_conv_channels, in_planes = 64):
    out_channels = [in_planes for i in range(n_conv_channels-1)]
    out_channels.append(out_channel)
    print(dict.keys())
    for i in range(n_conv_channels):
        prefix_key = f'tokenizer.conv_layers.{i}.conv'
        if old_sr == 1.0:
            full_weight = dict[f'{prefix_key}.weight']
            dict.pop(f'{prefix_key}.weight')
        else:
            full_weight = torch.cat((dict[f'{prefix_key}.conv.weight'], dict[f'{prefix_key}.AdaFM_Conv.conv_t.weight']), dim = 0)
            dict.pop(f'{prefix_key}.conv.weight')
            dict.pop(f'{prefix_key}.AdaFM_Conv.conv_t.weight')
        shared_channels = int(math.floor(new_sr*out_channels[i]))
        if new_sr == 1.0:
            dict[f'{prefix_key}.weight'] = copy.deepcopy(full_weight)
        else:
            dict[f'{prefix_key}.conv.weight'] = copy.deepcopy(full_weight[:shared_channels])
            dict[f'{prefix_key}.AdaFM_Conv.conv_t.weight'] = copy.deepcopy(full_weight[shared_channels:])
    print(f"Changed model_dict to support for split_ratio {new_sr} from {old_sr}")


#######################################################################################################################################
args = args()

task_parent_dict = {}

criterion = SoftTargetCrossEntropy().cuda()
criterion2 = nn.CrossEntropyLoss().cuda()
criterion3 = nn.MSELoss().cuda()
set_seed(3473)
if args.dataset != 'imagenet100':
    torch.cuda.set_device(model_args.gpu_no)


def compute_importance(train_loader, task, model, task_ord_list, task_id=-1):
    model.eval()
    imp_params = []
    name_list = []

    

    importance_val = 0.0
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        if args.dataset == '5d':
            inputs = inputs[0]
        inputs, targets = inputs.cuda(), targets.cuda()
        outputs, features =model(inputs, task_id, task_ord_list)



        

        importance_val += dist.item()



    print("TASK ID = {}, Importance = {}".format(task_id, importance_val))

    return importance_val




def run_model(split_ratio = 1.0, ker_sz = 9, dilation = 1, epochs = 250):

    print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
    print(f"Starting Checkpoint")

    print(f"Tokenizer Split Ratio is {split_ratio}:{round(1.0-split_ratio, 1)}")
    print(f"Conv_mtx kernel size is {ker_sz}X{ker_sz}")
    print(f"No of epochs is {epochs}")
    

    if args.dataset == '5d':
        print("FIVE DATASETS")
        data, taskcla,inputsize = five_loader.get(pc_valid=0.00)
    else:
        inc_dataset = datal.IncrementalDataset(
                                    dataset_name=args.dataset,
                                    args = args,
                                    random_order=args.random_classes,
                                    shuffle=True,
                                    seed=1,
                                    batch_size=args.train_batch,
                                    workers=args.workers,
                                    validation_split=args.validation,
                                    increment=args.class_per_task,
                                )

    task_model=[]
    task_acc=[]

    


    mixup_args = {
        'mixup_alpha': 0.3,
        'cutmix_alpha': 0.3,
        'cutmix_minmax': None,
        'prob': 0.7,
        'switch_prob': 0.5,
        'mode': 'elem',
        'label_smoothing': 0.5,
        'num_classes': args.class_per_task}


    modelm = models.__dict__[args.cct_val](img_size=args.inp_size,
                                        num_classes=args.class_per_task,
                                        positional_embedding='sine',
                                        n_conv_layers=args.n_conv_layers,
                                        kernel_size=args.tok_kernel_size,
                                        split_ratio = split_ratio,
                                        ker_sz = ker_sz,
                                        dilation = dilation,
                                        recon_ker_size=5, recon_ratio=0.5).cuda()

    task_done = 0
    # 
    global task_parent_dict
    task_parent_dict = {}
    if model_args.use_saved:
        if is_task0:
            if args.dataset != 'imagenet100':
                modelm.load_state_dict(torch.load(task0_model_path))
            else:
                model_dict = remove_module_from_state_dict(task0_model_path)
                modelm.load_state_dict(model_dict)
        else:
            model_dict = torch.load(task0_model_path)
            if args.dataset == 'imagenet100':
                model_dict = remove_module_from_state_dict(task0_model_path)
            change_tokenizer_state(model_dict, 1.0, split_ratio, 256, args.n_conv_layers)
            modelm.load_state_dict(model_dict)
            modelm.load_params(f'saved_model_parts/{model_args.dataset}')
            print('Loading from', f'saved_model_parts/{model_args.dataset}')
            with open(f'saved_model_parts/{model_args.dataset}/task_parent_dict.pkl', 'rb') as fid:
                task_parent_dict =  pickle.load(fid)
            task_done = modelm.get_task()
            print(task_done, task_parent_dict)

    test_loader_list = []
    for task in range(args.num_task):
        if is_task0 and task:
            break
        mixup_fn = Mixup(**mixup_args)

        if True:
            best_acc=0
            print('Training Task :---'+str(task))
            if args.dataset == '5d':
                train_loader, test_loader = get_five_loaders(data, task, args.train_batch, args.train_batch)
            else:
                task_info, train_loader, val_loader, test_loader = inc_dataset.new_task()
            test_loader_list.append(test_loader)

            lr = .05  # learning rate



            if task >= task_done:
                if task > 0:
                    print("Calculating Similarity for task {}".format(task))
                    if True:
                        task_parent_dict[task] = 0

                        task_ord_list = get_task_ord_list(task_parent_dict , task)


                        modelm.set_parent(False)
                    else:
                        task_parent_dict[task] = task
                        modelm.set_parent(True)
                else:
                    task_parent_dict[task] = task
                    task_ord_list = []
                    modelm.set_parent(True)
                print("[Task order List: {}]".format(task_ord_list))
                if task or is_task0:
                    modelm = train(train_loader, epochs, task, modelm, task_ord_list, mixup_fn, test_loader, split_ratio)
                else:
                    model_dict = torch.load(task0_model_path)
                    if args.dataset == 'imagenet100':
                        model_dict = remove_module_from_state_dict(task0_model_path)
                    modelm.load_state_dict(model_dict, strict=False)

                if is_task0:
                    torch.save(modelm.state_dict(), task0_model_path)
                    print(f"Saving..... for task{task}")
                modelm.update_global(task)
                if not is_task0:
                    modelm.save_params(f'saved_model_parts/{model_args.dataset}')
                    with open(f'saved_models/{model_args.dataset}/task_parent_dict.pkl', 'wb') as fid:
                        pickle.dump(task_parent_dict, fid)
                    torch.save(modelm.state_dict(), f'saved_model_parts/{model_args.dataset}/state_dict/state_dict.pkl')

                
            test_all(test_loader_list, modelm, task_parent_dict)


    


    print(f"Ending Checkpoint")
    print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
    


run_model(split_ratio=model_args.sr, ker_sz=model_args.ker_sz, dilation=model_args.dil, epochs=model_args.nepochs)
