import os
import random
import math
import shutil
import torch
import torch.nn as nn
import torchvision.transforms as transform
import torch.autograd.functional as F

########################################## 1 ################################################
def data_split(old_path):
    """copy data from path and organize three dirs
    test_set: ./dog/files, ./cat/files
    train_set: ./dog/files, ./cat/files
    val_set: ./dog/files, ./cat/files
    """
    new_path = 'data'
    if os.path.exists('data') == 0:
        os.makedirs(new_path)
    for root_dir, sub_dirs, file in os.walk(old_path):                               # 遍历os.walk(）返回的每一个三元组，内容分别放在三个变量中
        for sub_dir in sub_dirs:
            file_names = os.listdir(os.path.join(root_dir, sub_dir))                 # 遍历每个次级目录
            file_names = list(filter(lambda x: x.endswith('.jpg'), file_names))      # 去掉列表中的非jpg格式的文件

            random.shuffle(file_names)
            for i in range(len(file_names)):
                if i < math.floor(0.8*len(file_names)):
                    sub_path = os.path.join(new_path, 'train_set', sub_dir)
                elif i < math.floor(0.9*len(file_names)):
                    sub_path = os.path.join(new_path, 'val_set', sub_dir)
                elif i < len(file_names):
                    sub_path = os.path.join(new_path, 'test_set', sub_dir)
                if os.path.exists(sub_path) == 0:
                    os.makedirs(sub_path)

                shutil.copy(os.path.join(root_dir, sub_dir, file_names[i]), os.path.join(sub_path, file_names[i]))   # 复制图片，从源到目的地


data_path = 'old_data'
data_split(data_path)

class DataSet(Dataset):
    def __init__(self, data_path):  # 除了这两个参数之外，还可以设置其它参数
        self.label_name = {'ants': 0, 'bees': 1}
        self.data_info = get_info(data_path)

    def __getitem__(self, index):
        label, img_path = self.data_info[index]
        pil_img = Image.open(img_path).convert('RGB')  # 读数据
        re_img = transforms.Resize((32, 32))(pil_img)
        img = transforms.ToTensor()(re_img)  # PIL转张量
        return img, label

    def __len__(self):
        return len(self.data_info)

def get_info(data_path):
    data_info = list()
    for root_dir, sub_dirs, _ in os.walk(data_path):
        for sub_dir in sub_dirs:
            file_names = os.listdir(os.path.join(root_dir, sub_dir))
            img_names = list(filter(lambda x: x.endswith('.jpg'), file_names))
            for i in range(len(img_names)):
                img_path = os.path.join(root_dir, sub_dir, img_names[i])
                img_label = label_name[sub_dir]
                data_info.append((img_label, img_path))

    return data_info
train_set_path = os.path.join('data', 'train_set')
val_set_path = os.path.join('data', 'val_set')
test_set_path = os.path.join('data', 'test_set')
train_set = DataSet(data_path=train_set_path)
val_set = DataSet(data_path=val_set_path)
test_set = DataSet(data_path=test_set_path)
######################################### 1 #################################################


######################################## 2 ##################################################

def data_list(path):
    """copy data from path and organize three dirs
    test_file.txt: ./dog/files, ./cat/files
    train_file.txt: ./dog/files, ./cat/files
    val_file.txt: ./dog/files, ./cat/files
    """
    for root_dir, sub_dirs, _ in os.walk(path):                               # 遍历os.walk(）返回的每一个三元组，内容分别放在三个变量中
        idx = 0
        for sub_dir in sub_dirs:
            file_names = os.listdir(os.path.join(root_dir, sub_dir))                 # 遍历每个次级目录
            file_names = list(filter(lambda x: x.endswith('.jpg'), file_names))      # 去掉列表中的非jpg格式的文件

            random.shuffle(file_names)
            for i in range(len(file_names)):
                if i < math.floor(0.8 * len(file_names)):
                    txt_name = 'train_set.txt'
                elif i < math.floor(0.9 * len(file_names)):
                    txt_name = 'val_set.txt'
                elif i < len(file_names):
                    txt_name = 'test_set.txt'
                with open(os.path.join(path, txt_name), mode='a') as file:
                    file.write(str(idx) + ',' + os.path.join(path, sub_dir, file_names[i]) + '\n')     # 为了以后好用，修改了这里，将' '改成了','，另外路径加了sub_dir
            idx += 1

data_path = 'old_data'
data_list(data_path)


class DataSet(Dataset):
    def __init__(self, data_path):  # 除了这两个参数之外，还可以设置其它参数
        self.label_name = {'ants': 0, 'bees': 1}
        self.data_info = get_info_list(data_path)

    def __getitem__(self, index):
        label, img_path = self.data_info[index]
        pil_img = Image.open(img_path).convert('RGB')  # 读数据
        re_img = transforms.Resize((32, 32))(pil_img)
        img = transforms.ToTensor()(re_img)  # PIL转张量
        return img, label

    def __len__(self):
        return len(self.data_info)

def get_info_list(list_path):
    data_info = list()
    with open(list_path, mode='r') as f:
        lines = f.readlines()
        for i in range(len(lines)):
            img_label = int(lines[i].split(' ')[0])
            img_path = lines[i].split(' ')[1]
            data_info.append((img_label, img_path))
    return data_info

train_set_path = os.path.join('data', 'train_set')
val_set_path = os.path.join('data', 'val_set')
test_set_path = os.path.join('data', 'test_set')
train_set = DataSet(data_path=train_set_path)
val_set = DataSet(data_path=val_set_path)
test_set = DataSet(data_path=test_set_path)
########################################## 2 ################################################


########################################## 3 ################################################
import torch
import random
import time
from torchvision import datasets
from torchvision import transforms
import numpy as np

def get_std_mean(dataset, ratio=1):
    """get the mean and var of a dataset by sampling
    mean and var are used to Normalize dataset
    ratio: sampling ratio
    """
    data_x = dataset.data
    data_x = torch.transpose(torch.from_numpy(data_x), dim0=1, dim1=3) #(50000,32,32)->(50000,3,32,32)
    data_num = len(data_x)
    idx = list(range(data_num))
    random.shuffle(idx)                                                #产生随机索引
    data_selected = data_x[idx[0:int(ratio * data_num)]]
    mean = np.mean(data_selected.numpy(), axis=(0, 2, 3)) / 255
    std = np.std(data_selected.numpy(), axis=(0, 2, 3)) / 255
    return mean, std

def get_mean_std(dataset, ratio=1):

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=int(len(dataset)*ratio),
                                             shuffle=True, num_workers=0)
    data = iter(dataloader).next()[0]   # 一个batch的数据

    mean = np.mean(data.numpy(), axis=(0,2,3))
    std = np.std(data.numpy(), axis=(0,2,3))
    return mean, std

def get_std_mean(dataset, ratio=1):
    data_x = dataset.data
    data_x = torch.transpose(torch.from_numpy(data_x), dim0=1, dim1=3)  # (50000,32,32)->(50000,3,32,32)
    data_num = len(data_x)
    idx = list(range(data_num))
    random.shuffle(idx)  # 产生随机索引
    data_selected = data_x[idx[0:int(ratio * data_num)]]
    mean = np.mean(data_selected.numpy(), axis=(0, 2, 3)) / 255
    std = np.std(data_selected.numpy(), axis=(0, 2, 3)) / 255
    return mean, std

if __name__ == '__main__':

    train_dataset = datasets.CIFAR10('./data',
                                     train=True, download=False,
                                     transform=transforms.ToTensor())

    test_dataset = datasets.CIFAR10('./data',
                                    train=False, download=False,
                                    transform=transforms.ToTensor()

    time0 = time.time()
    train_mean, train_std = get_std_mean(train_dataset)
    test_mean, test_std = get_std_mean(test_dataset)
    time1 = time.time()
    time = time1 - time0
########################################## 3 ################################################

########################################## 4 ################################################

import functools
from functools import wraps
from time import time

def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        print(f'func: {f.__name__} args:{args, kw} took: {te-ts} sec')
        return result
    return wrap
########################################## 4 ################################################

