# -*- encoding: utf-8 -*-
'''
file       :FFOSMLDGTrainer.py
Author     :
'''

import numpy as np
import copy
import torch
import random
from torch.func import functional_call
import utils

class FFOSMLDGTrainer(object):
    def __init__(self,
                 model,
                 optimizer, # 这个优化器是元优化器 (meta-optimizer)
                 inner_lr: float = 1e-3,
                 max_iter: int = 1000,
                 log_interval: int = 1,
                 early_stop: int = 0,
                 device: str = "cuda:0",
                 **kwargs):
        """
        Initializes the Fast First-Order Sequential MLDG Trainer.
        """
        super(FFOSMLDGTrainer, self).__init__()

        self.model = model.to(device)
        self.optimizer = optimizer # 元优化器
        self.inner_lr = inner_lr   # 内循环学习率 (α)
        self.max_iter = max_iter
        self.log_interval = log_interval
        self.early_stop = early_stop
        self.device = device
        self.best_model_state = None

    def get_model_state(self):
        """Get the current state of the model."""
        return self.model.state_dict()

    def get_best_model_state(self):
        """Get the best state of the model."""
        return self.best_model_state

    def train(self, source_loaders, target_loader):
        """
        使用 Fast First-Order S-MLDG (FFO-S-MLDG) 算法进行训练。
        """
        stop = 0
        best_acc = 0.0
        log = []

        for iteration in range(self.max_iter):
            self.model.train()

            # 对应 Algorithm 2: θ̃ = θ
            # 1. 复制模型的原始参数，作为内循环的起点 (θ̃)
            fast_weights = {name: param.clone() for name, param in self.model.named_parameters()}

            # 2. 将源域的顺序随机打乱
            # 对应 Algorithm 2: p = shuffle([1, 2, ..., N])
            shuffled_loaders = random.sample(source_loaders, len(source_loaders))

            # 3. 内循环 (Inner-loop update)
            # 按照随机顺序，依次在每个源域上更新 fast_weights
            inner_loss = torch.tensor(0.0) # 用于记录日志
            for source_loader in shuffled_loaders:
                # 从当前域获取一个批次的数据
                try:
                    src_x, src_y = next(iter(source_loader))
                except ValueError:
                    # 兼容返回3个值的情况
                    src_x, src_y, _ = next(iter(source_loader))

                src_x, src_y = src_x.to(self.device), src_y.to(self.device)

                # 使用 fast_weights 在当前域的数据上计算损失
                loss = functional_call(self.model, fast_weights, (src_x, src_y))

                # 计算相对于 fast_weights 的梯度
                grads = torch.autograd.grad(loss, list(fast_weights.values()))

                # 手动更新 fast_weights: θ̃ = θ̃ - α * ∇L
                fast_weights = {
                    name: p - self.inner_lr * g
                    for (name, p), g in zip(fast_weights.items(), grads)
                }
                inner_loss = loss # 保存最后一个内循环的损失用于日志

            # 4. 元更新 (Meta update)
            # 对应 Algorithm 2: θ := θ + γ(θ̃ - θ)
            self.optimizer.zero_grad()
            with torch.no_grad():
                for name, param in self.model.named_parameters():
                    # 计算 "元梯度" (meta-gradient)。
                    # 优化器执行 θ_new = θ - lr * grad
                    # 我们设置 grad = θ - θ̃，所以更新后为:
                    # θ_new = θ - lr * (θ - θ̃)
                    # 这等价于论文中的插值更新，将原始参数向内循环结束后的参数方向移动一小步
                    param.grad = param.data - fast_weights[name]

            self.optimizer.step()

            # 5. 评估和日志记录
            self.model.eval()
            with torch.no_grad():
                target_acc = self.test(target_loader)

            stop += 1
            if target_acc > best_acc:
                best_acc = target_acc
                self.best_model_state = copy.deepcopy(self.model.state_dict())
                stop = 0

            log.append([iteration, inner_loss.item(), target_acc, best_acc])

            info = (
                f'Iter: [{iteration + 1:2d}/{self.max_iter}], '
                f'Inner Loss: {inner_loss.item():.4f}, '
                f'Target Acc: {target_acc:.4f}, '
                f'Best Acc: {best_acc:.4f}'
            )

            if (self.early_stop > 0 and stop >= self.early_stop):
                print(info)
                break

            if (iteration + 1) % self.log_interval == 0 or iteration == 0:
                print(info)

        np_log = np.array(log, dtype=float)
        return best_acc, np_log

    def test(self, dataloader):
        # 这个 test 方法与 LLLTrainer 中的实现一致
        try:
            # 兼容 PMDataset
            feature, labels, _ = dataloader.dataset.get_data()
        except:
            # 兼容 BaseDataset
            feature = dataloader.dataset.data()
            labels = dataloader.dataset.label()

        labels = np.argmax(labels.cpu().numpy(), axis=1)
        y_preds = self.model.predict(feature.to(self.device))
        acc = np.sum(y_preds == labels) / len(labels)
        return acc * 100.