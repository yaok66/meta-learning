# -*- encoding: utf-8 -*-
'''
file       :ContrastiveLoss.py
Date       :2025/03/10 10:11:40
Email      :qiang.wang@stu.xidian.edu.cn
Author     :qwangxdu
'''
import torch
from torch import nn
import torch.nn.functional as F

# class ContrastiveLoss(nn.Module):
#     def __init__(self, margin=1.0):
#         super(ContrastiveLoss, self).__init__()
#         self.margin = margin

#     def get_cos_similarity_distance(self, features):
#         """Get distance in cosine similarity
#         :param features: features of samples, (batch_size, num_clusters)
#         :return: distance matrix between features, (batch_size, batch_size)
#         """
#         # (batch_size, num_clusters)
#         features_norm = torch.norm(features, dim=1, keepdim=True)
#         # (batch_size, num_clusters)
#         features = features / features_norm
#         # (batch_size, batch_size)
#         cos_dist_matrix = torch.mm(features, features.transpose(0, 1))
#         return cos_dist_matrix

#     def forward(self, features, labels):
#         device = features.device
#         cos_dist_matrix = self.get_cos_similarity_distance(features)
#         cos_dist_matrix = (cos_dist_matrix + 1) / 2
#         labels = labels.view(-1, 1)
#         mask = torch.eq(labels, labels.T).float().to(device)
#         positive_loss = (mask * cos_dist_matrix).mean()
#         # print(positive_loss)
#         positive_loss = torch.tensor(0).to(device)
#         return positive_loss 

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, features, labels):
        # print(features.shape, labels.shape)
        # 计算样本间的欧几里得距离
        distance_matrix = torch.cdist(features, features, p=2)
        labels = labels.view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(features.device)

        # 计算对比损失
        positive_loss = (mask * distance_matrix ** 2).mean()
        negative_loss = ((1 - mask) * F.relu(self.margin - distance_matrix) ** 2).mean()

        loss = positive_loss + negative_loss
        return loss

    
class SupConLoss(nn.Module): 
    def __init__(self, temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        device = features.device
        features = F.normalize(features, dim=1)
        similarity_matrix = torch.exp(torch.mm(features, features.T) / self.temperature)

        labels = labels.view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)

        log_prob = torch.log(similarity_matrix / similarity_matrix.sum(dim=1, keepdim=True))
        loss = - (mask * log_prob).sum(dim=1) / mask.sum(dim=1)
        return loss.mean()
