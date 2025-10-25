# -*- encoding: utf-8 -*-
import numpy as np
import copy
import torch
import random
import utils

# from MLDG import linear 

class MLDGTrainer(object):
    def __init__(self,
                 model,
                 optimizer,
                 inner_lr: float = 1e-3,
                 meta_val_beta: float = 1.0,
                 max_iter: int = 1000,
                 log_interval: int = 1,
                 early_stop: int = 0,
                 device: str = "cuda:0",
                 stop_gradient: bool = False,
                 meta_step_size: float = 1e-3,
                 **kwargs):
        super(MLDGTrainer, self).__init__()
        self.model = model.to(device)
        self.optimizer = optimizer
        self.inner_lr = inner_lr
        self.meta_val_beta = meta_val_beta
        self.max_iter = max_iter
        self.log_interval = log_interval
        self.early_stop = early_stop
        self.device = device
        self.best_model_state = None
        
        self.stop_gradient = stop_gradient
        self.meta_step_size = meta_step_size


    def get_model_state(self):
        return self.model.state_dict()

    def get_best_model_state(self):
        return self.best_model_state

    def train(self, source_loaders, target_loader):
        stop = 0
        best_acc = -1.0
        log = []
        source_indices = list(range(len(source_loaders)))

        for iteration in range(self.max_iter):
            # --- 设置模型训练模式
            self.model.train()

            # --- 随机选择一个源域数据源作为 meta-val 集 ---
            val_idx = random.choice(source_indices)
            meta_val_loader = source_loaders[val_idx]
            # --- 获取 meta-train 集 ---
            meta_train_loaders = [loader for i, loader in enumerate(source_loaders) if i != val_idx]

            meta_train_loss = torch.tensor(0.0, device=self.device)
            # --- 内循环：累加 meta-train 损失 ---
            for train_loader in meta_train_loaders:
                try:
                    src_x, src_y = next(iter(train_loader))
                except ValueError:
                    src_x, src_y, _ = next(iter(train_loader))
                # 将src_x, src_y 转到GPU运行
                src_x, src_y = src_x.to(self.device), src_y.to(self.device)
                
                # feature = self.model.feature_extractor(src_x)
                # output = self.model.classifier(feature)
                # loss = self.criterion(output, src_y)
                # 将数据传入模型，求解损失
                loss = self.model(src_x, src_y)
                meta_train_loss = meta_train_loss + loss

            # --- 计算 meta-val 损失 ---
            try:
                val_x, val_y = next(iter(meta_val_loader))
            except ValueError:
                val_x, val_y, _ = next(iter(meta_val_loader))
            val_x, val_y = val_x.to(self.device), val_y.to(self.device)
            
            # Outer Update Loss
            meta_val_loss = self.model(val_x, val_y, 
                                       meta_loss = meta_train_loss, 
                                       meta_step_size = self.meta_step_size,
                                       stop_gradient = self.stop_gradient)
            
            # f: y = w * x + b # 默认w, b 存在在model中
            # funtioncal_call(f, (w1, b1), (x)) # 暂时将模型中的w,b临时替换掉，但是不更新
            
            # --- 组合总损失并优化 ---
            total_loss = meta_train_loss + self.meta_val_beta * meta_val_loss

            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()

            # --- 评估和日志记录 ---
            self.model.eval()
            with torch.no_grad():
                target_acc = self.test(target_loader)
            
            stop += 1
            if target_acc > best_acc:
                best_acc = target_acc
                self.best_model_state = copy.deepcopy(self.model.state_dict())
                stop = 0

            log.append([iteration, meta_train_loss.item(), meta_val_loss.item(), target_acc, best_acc])
            
            info = (f'Iter: [{iteration + 1:2d}/{self.max_iter}], '
                    f'MetaTrain Loss: {meta_train_loss.item():.4f}, '
                    f'MetaVal Loss: {meta_val_loss.item():.4f}, '
                    f'Target Acc: {target_acc:.4f}, '
                    f'Best Acc: {best_acc:.4f}')

            if (self.early_stop > 0 and stop >= self.early_stop):
                print(info); break
            if (iteration + 1) % self.log_interval == 0 or iteration == 0:
                print(info)

        return best_acc, np.array(log, dtype=float)

    def test(self, dataloader):
        feature = dataloader.dataset.data()
        labels = dataloader.dataset.label()
        
        labels = np.argmax(labels.cpu().numpy(), axis=1)
        y_preds = self.model.predict(feature.to(self.device))
        acc = np.sum(y_preds == labels) / len(labels)
        return acc * 100.