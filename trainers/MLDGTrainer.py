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

class MLDGTrainer(object):
    def __init__(self,
                 model,
                 optimizer, # 这个优化器是元优化器 (meta-optimizer)
                 inner_lr: float = 1e-3,
                 max_iter: int = 1000,
                 log_interval: int = 1,
                 early_stop: int = 0,
                 device: str = "cuda:0",
                 meta_val_beta: float = 0.0001,
                 **kwargs):
        """
        Initializes the Fast First-Order Sequential MLDG Trainer.
        """
        super(MLDGTrainer, self).__init__()

        self.model = model.to(device)
        self.optimizer = optimizer # 元优化器
        self.inner_lr = inner_lr   # 内循环学习率 (α)
        self.max_iter = max_iter
        self.log_interval = log_interval
        self.early_stop = early_stop
        self.device = device
        self.best_model_state = None
        self.meta_val_beta = meta_val_beta


    def get_model_state(self):
        """Get the current state of the model."""
        return self.model.state_dict()

    def get_best_model_state(self):
        """Get the best state of the model."""
        return self.best_model_state

    def train(self, source_loaders, target_loader):
        """
        使用 MLDG (Meta-Learning Domain Generalization) 算法进行训练。
        """
        stop = 0
        best_acc = -1.0
        log = []

        # 获取所有源域的索引
        source_indices = list(range(len(source_loaders)))

        # 外循环：整个训练的迭代
        for iteration in range(self.max_iter):
            self.model.train()

            # 1. 随机选择一个域作为 meta-val (元验证) 域
            val_idx = random.choice(source_indices)
            meta_val_loader = source_loaders[val_idx]

            # 2. 内循环：累加所有 meta-train (元训练) 域的损失
            meta_train_loss = torch.tensor(0.0, device=self.device)
            meta_train_loaders = [loader for i, loader in enumerate(source_loaders) if i != val_idx]

            for train_loader in meta_train_loaders:
                # 从每个元训练域获取数据
                try:
                    src_x, src_y = next(iter(train_loader))
                except ValueError:
                    src_x, src_y, _ = next(iter(train_loader))
                

                # forward with the adapted parameters
                src_x, src_y = src_x.to(self.device), src_y.to(self.device)
                

                # 直接在原始模型上计算损失并累加
                loss = self.model(src_x, src_y)
                meta_train_loss = meta_train_loss + loss

            # 3. 模拟内循环更新，并在 meta-test 域上计算损失
            
            # 3.1. 计算 meta_train_loss 对当前模型参数的梯度
            # 这是模拟内循环更新所需要的方向
            params = [p for p in self.model.parameters() if p.requires_grad]
            grads = torch.autograd.grad(meta_train_loss, params, create_graph=True)
            
            # 3.2. 计算更新一次后的临时权重 (fast_weights)
            fast_weights = {
                name: p - self.inner_lr * g
                for (name, p), g in zip(self.model.named_parameters(), grads)
            }
            
            # 3.3. 在 meta-test 域上计算验证损失 (meta_val_loss)
            try:
                val_x, val_y = next(iter(meta_val_loader))
            except ValueError:
                val_x, val_y, _ = next(iter(meta_val_loader))
            val_x, val_y = val_x.to(self.device), val_y.to(self.device)
            
            # 使用 functional_call 和 fast_weights 计算验证损失
            meta_val_loss = functional_call(self.model, fast_weights, (val_x, val_y))

            # 4. 组合总损失并执行一次优化
            total_loss = meta_train_loss + self.meta_val_beta * meta_val_loss

            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()

            # 5. 评估和日志记录 (与之前类似)
            self.model.eval()
            with torch.no_grad():
                target_acc = self.test(target_loader)
            
            stop += 1
            if target_acc > best_acc:
                best_acc = target_acc
                self.best_model_state = copy.deepcopy(self.model.state_dict())
                stop = 0

            log.append([iteration, meta_train_loss.item(), meta_val_loss.item(), target_acc, best_acc])
            
            info = (
                f'Iter: [{iteration + 1:2d}/{self.max_iter}], '
                f'MetaTrain Loss: {meta_train_loss.item():.4f}, '
                f'MetaVal Loss: {meta_val_loss.item():.4f}, '
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
        feature, labels, clusters = dataloader.dataset.get_data()
        labels = np.argmax(labels.numpy(), axis=1)
        y_preds = self.model.predict(feature.to(self.device))
        acc = np.sum(y_preds == labels) / len(labels)
        return acc * 100.