# -*- encoding: utf-8 -*-
'''
file       :LLLTrainer.py
Date       :2025/05/28 20:57:11
Email      :qiang.wang@stu.xidian.edu.cn
Author     :qwangxdu
'''

import numpy as np
import copy 
import torch
import utils
import random
from torch.func import functional_call

class LLLTrainer(object):
    def __init__(self, 
                 model, 
                 optimizer,
                 max_iter: int = 1000,
                 log_interval: int = 1, 
                 early_stop: int = 0,
                 device: str = "cuda:0", 
                 **kwargs):
        """
        Initializes the MetaTrainer.

        Parameters:
            model (torch.nn.Module): The model to be trained.
            optimizer (torch.optim.Optimizer): The optimizer for meta-training.
            log_interval (int, optional): Interval for logging training progress. Defaults to 1.
            early_stop (int, optional): Number of epochs without improvement before stopping. Defaults to 0.
            device (str, optional): Device to use for training. Defaults to "cuda:0".
        """
        super(LLLTrainer, self).__init__()
        
        # Initialize model and optimizers
        self.model = model.to(device)
        self.optimizer = optimizer

        # Training parameters
        self.max_iter = max_iter
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
            # 根据iteration按照顺序取源域
            idx = iteration % len_source_loaders
            cur_source_loader = source_loaders[iteration % len_source_loaders]

            source_iter = iter(cur_source_loader)
            try:
                src_x, src_y = next(source_iter)
            except StopIteration:
                source_iter = iter(cur_source_loader)
                src_x, src_y = next(source_iter)

            src_x, src_y = src_x.to(self.device), src_y.to(self.device)

            loss = self.model(src_x, src_y)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.model.eval()
            with torch.no_grad():
                # source_acc = self.test(source_loader)
                source_acc = self.test(cur_source_loader)
                target_acc = self.test(target_loader)
            log.append([iteration, loss.item(), source_acc, target_acc, best_acc])

            # TODO 记录最佳结果
            stop += 1
            if target_acc > best_acc:
                best_acc = target_acc
                self.best_model_state = copy.deepcopy(self.model.state_dict())
                stop = 0    

            # 保留信息，并在训练过程中输出
            info = (
                f'iter: [{iteration + 1:2d}/{self.max_iter}], '
                f'loss: {loss.item():.4f}, '
                f'source_acc: {source_acc:.4f}, '
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
    