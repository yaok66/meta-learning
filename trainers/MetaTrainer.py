# -*- encoding: utf-8 -*-
'''
file       :MetaTrainer.py
Date       :2025/05/28 10:55:47
Email      :qiang.wang@stu.xidian.edu.cn
Author     :qwangxdu
'''

import numpy as np
import copy 
import torch
import utils
import random
from torch.func import functional_call

class MetaTrainer(object):
    def __init__(self, 
                 model, 
                 meta_optimizer, 
                 inner_lr: float = 1e-3,
                 max_iter: int = 1000,
                 n_inner_steps: int = 1,
                 log_interval: int = 1, 
                 early_stop: int = 0,
                 device: str = "cuda:0", 
                 **kwargs):
        """
        Initializes the MetaTrainer.

        Parameters:
            model (torch.nn.Module): The model to be trained.
            meta_optimizer (torch.optim.Optimizer): The optimizer for meta-training.
            inner_optimizer (torch.optim.Optimizer): The optimizer for inner loop training.
            n_inner_steps (int, optional): Number of inner loop steps. Defaults to 1.
            log_interval (int, optional): Interval for logging training progress. Defaults to 1.
            early_stop (int, optional): Number of epochs without improvement before stopping. Defaults to 0.
            device (str, optional): Device to use for training. Defaults to "cuda:0".
        """
        super(MetaTrainer, self).__init__()
        
        # Initialize model and optimizers
        self.model = model.to(device)
        self.meta_optimizer = meta_optimizer
        self.inner_lr = inner_lr

        # Training parameters
        self.max_iter = max_iter
        self.n_inner_steps = n_inner_steps
        self.log_interval = log_interval
        self.early_stop = early_stop
        self.device = device

        # Additional attributes
        self.device = device
        self.best_model_state = None

    def get_model_state(self):
        """Get the current state of the model."""
        return self.model.state_dict()

    def get_best_model_state(self):
        """Get the best state of the model."""
        return self.best_model_state

    def train(self, source_loaders, target_loader):
        stop = 0
        best_acc = 0.0
        log = []

        len_source_loaders = len(source_loaders)

        for iteration in range(self.max_iter):
            self.model.train()
            # 随机选择两个源域
            support_loader, query_loader = random.sample(source_loaders, 2)
            # print(support_loader)

            support_iter = iter(support_loader)
            query_iter = iter(query_loader)
            # 获取支持集和查询集的数据
            try:
                support_data, support_label = next(support_iter)
            except StopIteration:
                support_iter = iter(support_loader)
                support_data, support_label = next(support_iter)
            
            try:
                query_data, query_label = next(query_iter)
            except StopIteration:
                query_iter = iter(query_loader)
                query_data, query_label = next(query_iter)

            support_data, support_label = support_data.to(self.device), support_label.to(self.device)
            query_data, query_label = query_data.to(self.device), query_label.to(self.device)

            # 复制模型用于内部循环
            fast_weights = {name: param for name, param in self.model.named_parameters()}
            for _ in range(self.n_inner_steps):
                loss = functional_call(
                    self.model, fast_weights, (support_data, support_label)
                )
                grads = torch.autograd.grad(loss, fast_weights.values(), create_graph=True)
                fast_weights = {
                    name: param - self.inner_lr * grad
                    for (name, param), grad in zip(fast_weights.items(), grads)
                }
            
            # 在查询集上计算损失
            query_loss = functional_call(
                self.model, fast_weights, (query_data, query_label)
            )
            self.meta_optimizer.zero_grad()
            query_loss.backward()   
            self.meta_optimizer.step()

            self.model.eval()
            with torch.no_grad():
                # source_acc = self.test(source_loader)
                support_acc = self.test(support_loader)
                query_acc = self.test(query_loader)
                target_acc = self.test(target_loader)
            log.append([iteration, query_loss.item(), support_acc, query_acc, target_acc, best_acc])

            # TODO 记录最佳结果
            stop += 1
            if target_acc > best_acc:
                best_acc = target_acc
                self.best_model_state = copy.deepcopy(self.model.state_dict())
                stop = 0    

            # 保留信息，并在训练过程中输出
            info = (
                f'iter: [{iteration + 1:2d}/{self.max_iter}], '
                f'loss: {query_loss.item():.4f}, '
                f'support_acc: {support_acc:.4f}, '
                f'query_acc: {query_acc:.4f}, '
                f'target_acc: {target_acc:.4f}, '
                f'best_acc: {best_acc:.4f} '
            )

            # TODO 早停止
            if ( self.early_stop > 0 and stop >= self.early_stop ) or ( 100 - best_acc < 1e-3):
                print(info)
                break

            # TODO 输出日志
            if (iteration + 1) % self.log_interval == 0 or iteration == 0:
                print(info)

        np_log = np.array(log, dtype=float)
        return best_acc, np_log
            
    def test(self, dataloader):
        feature = dataloader.dataset.data()
        labels = dataloader.dataset.label()
        labels = np.argmax(labels.numpy(), axis=1)
        y_preds = self.model.predict(feature.to(self.device))
        acc = np.sum(y_preds == labels) / len(labels)
        return acc * 100.
    