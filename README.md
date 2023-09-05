## Exemplar-Free Continual Transformer with Convolutions [[Paper]](https://arxiv.org/pdf/2308.11357.pdf) [[Website]](https://cvir.github.io/projects/contracon)

This repository contains the implementation details of our Exemplar-Free Continual Transformer with Convolutions (ConTraCon) approach for continual learning with transformer backbone.

Anurag Roy, Vinay Verma, Sravan Voonna, Kripabandhu Ghosh, Saptarshi Ghosh, Abir Das, "Exemplar-Free Continual Transformer with Convolutions"\
 

If you use the codes and models from this repo, please cite our work. Thanks!

```
@InProceedings{roy_2023_ICCV,
    author    = {Roy, Anurag and Verma, Vinay and Voonna, Sravan and Ghosh, Kripabandhu and Ghosh, Saptarshi and Das, Abir},
    title     = {Exemplar-Free Continual Transformer with Convolutions},
    booktitle = {International Conference on Computer Vision (ICCV)},
    year      = {2023}
}
```

## Requirements
The code is written for python `3.8.16`, but should work for other version with some modifications.
```
pip install -r requirements.txt
```
## Data Preparation

1. Download the datasets to the root diretory, `Datasets`.
2. `CIFAR100` dataset will be automatically downloaded, while [`ImageNet100`, `TinyImageNet`] requires manual download.
3. Overview of dataset root diretory

    ```shell
    ├── cifar100
    │   └── cifar-100-python
    ├── tinyimagenet
    │   ├── tiny-imagenet-200
    │   ├── train
    │   ├── val
    │   └── test
    └── imagenet-100
        ├── imagenet-r
        ├── train_list.txt
        └── val_list.txt
    ```


**NOTE** -- After downloading and extracting the tinyimagenet dataset inside the `Datasets` folder, run 
```
python val_format.py
```
This is to change the way the test dataset is stored for tinyimagenet.


## Python script overview

`auto_run.py` - Contains the training and the inference code for the ConTraCon approach.

`src/*` - Contains the source code for the backbone transformer architecture and the convolutional task adaptation mechanisms.

`src/utils/model_parts.py` - Contains the task specific adaptation classes and functions.

`incremental_dataloader.py` - Contains the code for the dataloaders for different datasets.

### Key Parameters:
 `ker_sz`: kernel size of the convolution kernels which are applied on the key, query and value weight matrices of the MHSA layers  \
 `num_tasks` : Number of tasks to split the given dataset into. This will split the classes in the datasets equally among the tasks \
 `nepochs`: Number of training epochs for each task \
 `is_task0`: Denotes whether training the first task. For the first task, the entire backbone transformer is trained from scratch. \
 `use_saved`: Use saved weights and resume training from next task. For example, for a 10 task setup, if trained till task 2, you can resume training from task 3 by using this flag. If training on all tasks have completed, then this flag can be used for re-evaluation of the trained model. \
 `dataset`: Denotes the dataset.\
 `data_path`:  The path for the dataset. \
 `scenario`: Evaluation scenario. We have evaluated our models in two scenarios -- `til` (task incremental learning) and `cil` (class incremental learning).
 


### Training ConTraCon
- For training on `x%` labeled data scenario, the first task needs to be trained first. This can be done by using the  `--is_task0` flag.
- For training on subsequent tasks, run without the `-is_task0` flag.

### Sample Code to train ConTraCon

The code to train ConTraCon on the ImageNet-100 dataset is provided as follows:


1. Training the first task : 
```
python auto_run.py --ker_sz 15 --nepochs 500 --dataset imagenet100 --data_path ./Datasets/imagenet-100/ --num_tasks 10 --is_task0 --scenario til
```

2. Training the rest of the tasks:
```
python auto_run.py --ker_sz 15 --nepochs 500 --dataset imagenet100 --data_path ./Datasets/imagenet-100/ --num_tasks 10 --scenario til
```

## Sample Code to Evaluate ConTraCon

- To evaluate ConTraCon in the til setup, run:
```
python auto_run.py --ker_sz 15 --nepochs 500 --dataset imagenet100 --data_path ./Datasets/imagenet-100/ --num_tasks 10 --use_saved --scenario til
```
- To evaluate ConTraCon in the cil setup, run:
```
python auto_run.py --ker_sz 15 --nepochs 500 --dataset imagenet100 --data_path ./Datasets/imagenet-100/ --num_tasks 10 --use_saved --scenario cil
```

## Reference

The implementation reused some portions from [CCT](https://github.com/SHI-Labs/Compact-Transformers)[1].


1. Ali Hassani, Steven Walton, Nikhil Shah, Abulikemu Abuduweili, Jiachen Li, Humphrey Shi. "Escaping the Big Data Paradigm with Compact Transformers." Arxiv Preprint. 2021.
