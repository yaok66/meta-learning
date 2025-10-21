# -*- encoding: utf-8 -*-
'''
file       :pm_loader.py
Date       :2025/02/17 20:53:34
Email      :qiang.wang@stu.xidian.edu.cn
Author     :qwangxdu
'''

import torch
import numpy as np
from sklearn.cluster import DBSCAN
from torch.utils.data import TensorDataset, DataLoader
from collections import Counter
from ._get_dataset import *

class PMDataset(TensorDataset):
    def __init__(self, d1, d2, d3):
        super(PMDataset, self).__init__()
        self.d1 = d1
        self.d2 = d2
        self.d3 = d3

    def __len__(self):
        return len(self.d1)
    
    def __getitem__(self, idx):
        return self.d1[idx], self.d2[idx], self.d3[idx]
    
    def get_data(self):
        return self.d1, self.d2, self.d3

def get_pm_laoder(args, dataset, target, source_lists = None):
    data, one_hot_mat, subject_ids = dataset["data"], dataset["one_hot_mat"], dataset["groups"][:, 0]

    if source_lists is None:
        source_lists = list(range(1, args.num_of_subjects + 1))
        source_lists.remove(target)
    source_lists = np.array(source_lists)

    loader_target = process_target_domain(args, data[subject_ids==target], one_hot_mat[subject_ids==target])

    loader_source = process_source_domains(
        args, 
        data[np.isin(subject_ids, source_lists)], 
        one_hot_mat[np.isin(subject_ids, source_lists)], 
        subject_ids[np.isin(subject_ids, source_lists)])

    return loader_source, loader_target

def add_noise_to_labels(labels, noise_level=0.1):
    """
    Add noise to one-hot encoded labels by randomly flipping them.
    
    Args:
    - labels (np.ndarray): One-hot encoded labels.
    - noise_level (float): The probability of flipping each label.
    
    Returns:
    - np.ndarray: Noisy labels.
    """
    noisy_labels = labels.copy()
    num_classes = labels.shape[1]
    for i in range(labels.shape[0]):
        if np.random.rand() < noise_level:
            # Randomly select a new class different from the current one
            current_class = np.argmax(labels[i])
            new_class = np.random.choice([c for c in range(num_classes) if c != current_class])
            noisy_labels[i] = np.eye(num_classes)[new_class]
    return noisy_labels

def perform_clustering(args, X, y):
    """
    执行DBSCAN聚类，并过滤噪声点。
    """
    clustering = DBSCAN(eps=args.eps, min_samples=args.min_samples).fit(X)
    labels = clustering.labels_

    num_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    mask = labels != -1  # 过滤掉噪声点
    return num_clusters, X[mask], y[mask], labels[mask]

def process_target_domain(args, x_target, y_target):
    # print(EEG[target_mask].shape)
    n, x, y, c = perform_clustering(args, x_target, y_target)
    setattr(args, "num_of_t_clusters", n)
    target_features = torch.from_numpy(x).to(torch.float32)
    target_labels = torch.from_numpy(y)
    target_cluster = torch.from_numpy(c)
    target_dataset = PMDataset(target_features, target_labels, target_cluster)
    target_loader = DataLoader(
        dataset=target_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True
    )
    return target_loader

def process_source_domains(args, x_source, y_source, s_source, noisy_level = 0.0):
    cluster_results = {}
    min_clusters = float('inf')

    for subject_id in np.unique(s_source):
        mask = s_source == subject_id
        n, x, y, c = perform_clustering(args, x_source[mask], y_source[mask])
        cluster_results[subject_id] = (n, x, y, c)
        min_clusters = min(min_clusters, n)

    setattr(args, "num_of_s_clusters", min_clusters)
    source_datasets = []

    for subject_id, (n, x, y, c) in cluster_results.items():
        # 为了保证每一个source都具有相同的cluster数量，因此将具有多余cluster的数据过滤掉
        # Filters the data to ensure each subject has the same number of clusters.
        cluster_counts = Counter(c)
        top_clusters = [cluster for cluster, _ in cluster_counts.most_common(int(min_clusters))]
        mask = np.isin(c, top_clusters)

        source_features = torch.from_numpy(x[mask]).to(torch.float32)
        # source_labels = torch.from_numpy(add_noise_to_labels(y[mask], noisy_level)).type(torch.Tensor)
        source_labels = torch.from_numpy(y[mask]) # .type(torch.Tensor)
        
        # Reorders cluster IDs to ensure consistency.
        source_cluster = c[mask]
        unique_clusters = np.unique(source_cluster)
        cluster_mapping = {old_label: new_label for new_label, old_label in enumerate(unique_clusters)}

        source_cluster = torch.tensor([cluster_mapping[label] for label in source_cluster], dtype=torch.long)

        # Cleans noisy labels by assigning the most common label within each cluster.
        for cluster in unique_clusters:
            cluster_mask = source_cluster == cluster
            if cluster_mask.sum() > 0:
                most_common_label = torch.mode(torch.argmax(source_labels[cluster_mask], dim=1), 0)[0]
                source_labels[cluster_mask] = torch.eye(source_labels.size(1))[most_common_label]

        source_datasets.append(PMDataset(source_features, source_labels, source_cluster))

    source_loaders = [
        DataLoader(
            dataset=dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers
        ) for dataset in source_datasets
    ]

    # Reorders the source loaders based on the specified startup source.
    n = args.startup_source - 1 # 比如说是第14个
    source_loaders = [source_loaders[n]] + source_loaders[:n] + source_loaders[n+1:]

    return source_loaders



