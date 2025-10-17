from torch import nn
from torch.nn import init

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise ValueError('Boolean value expected.')
    

import torch
import torch.nn.functional as F
import numpy as np
from scipy.stats import entropy

def compute_kl_divergence(source_features, target_features, use_torch=True):
    """
    计算源域与目标域的 KL 散度，衡量它们的分布相似性。
    - source_features: (N, D) 形状的张量，表示源域的特征分布
    - target_features: (M, D) 形状的张量，表示目标域的特征分布
    - use_torch: 是否使用 PyTorch 计算 KL 散度
    """
    # 计算概率分布（可以用 Softmax 归一化）
    source_prob = F.softmax(source_features, dim=-1)  # (N, D)
    target_prob = F.softmax(target_features, dim=-1)  # (M, D)

    # 计算均值概率
    source_mean = torch.mean(source_prob, dim=0)  # (D,)
    target_mean = torch.mean(target_prob, dim=0)  # (D,)

    # 避免数值不稳定（加上一个小的数值）
    eps = 1e-10
    source_mean = torch.clamp(source_mean, eps, 1.0)
    target_mean = torch.clamp(target_mean, eps, 1.0)

    # 计算 KL 散度
    if use_torch:
        kl_div = F.kl_div(torch.log(source_mean), target_mean, reduction="sum").item()
    else:
        kl_div = entropy(source_mean.cpu().numpy(), target_mean.cpu().numpy())

    return kl_div

