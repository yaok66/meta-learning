# -*- encoding: utf-8 -*-
'''
file       :maxuppm_loader.py
Date       :2025/03/29 20:05:24
Email      :qiang.wang@stu.xidian.edu.cn
Author     :qwangxdu
'''

import torch
import numpy as np
from sklearn.cluster import DBSCAN
from torch.utils.data import TensorDataset, DataLoader
from collections import Counter
from ._get_dataset import *

from dataloaders.pm_loader import PMDataset, perform_clustering, process_target_domain


class MultiSourceDataset(TensorDataset):
    def __init__(self, source_domains):
        """
        :param source_domains: List of N dictionaries, each containing:
                               - "x": Tensor (n_i, F)
                               - "y": Tensor (n_i,)
                               - "cluster": Tensor (n_i,)
        """
        self.num_domains = len(source_domains)

        # 计算最小样本数
        self.min_samples = min(domain["x"].shape[0] for domain in source_domains)

        # 预处理数据，确保高效索引
        self.x_all = torch.cat([domain["x"][:self.min_samples] for domain in source_domains], dim=0)  # (num_domains * min_samples, F)
        self.y_all = torch.cat([domain["y"][:self.min_samples] for domain in source_domains], dim=0)  # (num_domains * min_samples,)
        self.cluster_all = torch.cat([domain["cluster"][:self.min_samples] for domain in source_domains], dim=0)  # (num_domains * min_samples,)

        # 重新组织成 (num_domains, min_samples, F) 的形状，便于索引
        self.x_all = self.x_all.view(self.num_domains, self.min_samples, -1)
        self.y_all = self.y_all.view(self.num_domains, self.min_samples, -1)
        self.cluster_all = self.cluster_all.view(self.num_domains, self.min_samples)

    def __len__(self):
        return self.min_samples

    def __getitem__(self, index):
        """
        Returns:
            x: (num_domains, F)
            y: (num_domains,)
            cluster: (num_domains,)
        """
        return self.x_all[:, index], self.y_all[:, index], self.cluster_all[:, index]

    def get_single_domain(self, domain_index):
        """
        获取单个源域的所有数据
        :return: (Tensor(min_samples, F), Tensor(min_samples,), Tensor(min_samples,))
        """
        if 0 <= domain_index < self.num_domains:
            return self.x_all[domain_index], self.y_all[domain_index], self.cluster_all[domain_index]
        raise IndexError(f"域索引需在 0~{self.num_domains-1} 之间")

    def get_data(self):
        """
        获取所有源域拼接后的数据
        :return: (Tensor(num_domains * min_samples, F)
        """
        return self.x_all.view(-1, self.x_all.shape[-1]), self.y_all.view(-1, self.y_all.shape[-1]), self.cluster_all.view(-1)
    
def get_maxuppm_laoder(args, dataset, target, source_lists = None):
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

def process_source_domains(args, x_source, y_source, s_source):
    cluster_results = {}
    min_clusters = float('inf')

    for subject_id in np.unique(s_source):
        mask = s_source == subject_id
        n, x, y, c = perform_clustering(args, x_source[mask], y_source[mask])
        cluster_results[subject_id] = (n, x, y, c)
        min_clusters = min(min_clusters, n)

    setattr(args, "num_of_s_clusters", min_clusters)
    source_domains = []
    for subject_id, (n, x, y, c) in cluster_results.items():
        # 为了保证每一个source都具有相同的cluster数量，因此将具有多余cluster的数据过滤掉
        # Filters the data to ensure each subject has the same number of clusters.
        cluster_counts = Counter(c)
        top_clusters = [cluster for cluster, _ in cluster_counts.most_common(int(min_clusters))]
        mask = np.isin(c, top_clusters)

        source_features = torch.from_numpy(x[mask]).to(torch.float32)
        # source_labels = torch.from_numpy(add_noise_to_labels(y[mask], noisy_level)).type(torch.Tensor)
        source_labels = torch.from_numpy(y[mask]).to(torch.float32)
        
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

        source_domains.append({
            "x": source_features,
            "y": source_labels,
            "cluster": source_cluster
        })
    
    source_datasets = MultiSourceDataset(source_domains)
    source_loader = DataLoader(
        dataset=source_datasets,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
    )
    return source_loader
