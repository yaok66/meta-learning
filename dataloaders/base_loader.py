# -*- encoding: utf-8 -*-
'''
file       :seed_loader.py
Date       :2025/02/17 17:27:11
Email      :qiang.wang@stu.xidian.edu.cn
Author     :qwangxdu
'''

import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from ._get_dataset import *

class BaseDataset(TensorDataset):
    def __init__(self, d1, d2):
        super(BaseDataset, self).__init__()
        self.d1 = d1
        self.d2 = d2

    def __len__(self):
        return len(self.d1)
    
    def __getitem__(self, idx):
        return self.d1[idx], self.d2[idx]
    
    def data(self):
        return self.d1
    def label(self):
        return self.d2

def get_base_laoder(args, dataset, target, source_lists = None):
    data, one_hot_mat, subject_ids = dataset["data"], dataset["one_hot_mat"], dataset["groups"][:, 0]

    if source_lists is None:
        source_lists = list(range(1, args.num_of_subjects + 1))
        source_lists.remove(target)
    source_lists = np.array(source_lists)

    # 获得目标域数据
    target_features = torch.from_numpy(data[subject_ids == target]).to(torch.float32)
    target_labels = torch.from_numpy(one_hot_mat[subject_ids == target])
    torch_dataset_target = BaseDataset(target_features, target_labels)
    
    # 获得源域数据
    source_features = torch.from_numpy(data[np.isin(subject_ids, source_lists)]).to(torch.float32)
    source_labels = torch.from_numpy(one_hot_mat[np.isin(subject_ids, source_lists)])
    torch_dataset_source = BaseDataset(source_features, source_labels)

    loader_source = DataLoader(
            dataset=torch_dataset_source,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers
            )
    loader_target = DataLoader(
            dataset=torch_dataset_target,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers
            )

    return loader_source, loader_target

def get_mutlisource_laoder(args, dataset, target, source_lists = None):
    data, one_hot_mat, subject_ids = dataset["data"], dataset["one_hot_mat"], dataset["groups"][:, 0]

    if source_lists is None:
        source_lists = list(range(1, args.num_of_subjects + 1))
        source_lists.remove(target)
    source_lists = np.array(source_lists)

    # 获得目标域数据
    target_features = torch.from_numpy(data[subject_ids == target]).to(torch.float32)
    target_labels = torch.from_numpy(one_hot_mat[subject_ids==target])
    torch_dataset_target = BaseDataset(target_features, target_labels)
    
    # 获得每个源域的数据，分别生成 DataLoader
    loader_source = []
    for src in source_lists:
        src_features = torch.from_numpy(data[subject_ids == src]).to(torch.float32)
        src_labels = torch.from_numpy(one_hot_mat[subject_ids == src])
        torch_dataset_source = BaseDataset(src_features, src_labels)
        loader = DataLoader(
            dataset=torch_dataset_source,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers
        )
        loader_source.append(loader)
        
    loader_target = DataLoader(
            dataset=torch_dataset_target,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers
            )

    return loader_source, loader_target

