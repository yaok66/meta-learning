# -*- encoding: utf-8 -*-
'''
file       :MaxUpLLL.py
Date       :2025/05/28 23:41:38
Email      :qiang.wang@stu.xidian.edu.cn
Author     :qwangxdu
'''

import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np

class FeatureExtractor(nn.Module):
    def __init__(self, input_dim=310, hidden_1=64, hidden_2=64):
        super(FeatureExtractor, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_1)
        self.fc2 = nn.Linear(hidden_1, hidden_2)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        return x

    def get_parameters(self):
        params = [
            {"params": self.fc1.parameters(), "lr_mult": 1},
            {"params": self.fc2.parameters(), "lr_mult": 1},
        ]
        return params

class LabelClassifier(nn.Module):
    def __init__(self, input_dim = 64, num_of_class=3):
        super(LabelClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, num_of_class)
    
    def forward(self, feature):
        y = self.fc1(feature)
        return self.fc2(y)
    
    def predict(self, feature):
        with torch.no_grad():
            logits = F.softmax(self.forward(feature), dim=1)
            y_preds = np.argmax(logits.cpu().numpy(), axis=1)
        return y_preds
    
    def get_parameters(self):
        params = [
            {"params": self.fc1.parameters(), "lr_mult": 1},
            {"params": self.fc2.parameters(), "lr_mult": 1}
        ]
        return params


class MaxUpLLL(nn.Module):
    def __init__(self, 
                 input_dim: int= 310, 
                 num_of_class: int=3, 
                 max_iter: int=1000, 
                 num_of_sources: int=14,
                 **kwargs):
        super(MaxUpLLL, self).__init__()

        self.feature_extractor = FeatureExtractor(input_dim=input_dim)
        self.classifier = LabelClassifier(input_dim=64, num_of_class=num_of_class)
        self.num_of_sources = num_of_sources
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, source, source_label):
        
        source = source.permute(1, 0, 2) # [N_sources, B, F] -> [B, N_sources, F]
        source_label = source_label.permute(1, 0, 2)

        # compute all source losses
        with torch.no_grad():
            source_all = source.reshape(-1, source.size(-1))
            source_label_all = source_label.reshape(-1, source_label.size(-1))
            source_feature = self.feature_extractor(source_all)
            source_clf = self.classifier(source_feature)
            loss_all = F.cross_entropy(source_clf, source_label_all, reduction='none')
            loss_chunks = loss_all.view(self.num_of_sources, -1)
            loss_per_chunk = loss_chunks.mean(dim=-1)

        max_loss_idx = torch.argmax(loss_per_chunk)

        max_source = source[max_loss_idx]
        max_source_label = source_label[max_loss_idx]
        max_source_feature = self.feature_extractor(max_source)

        max_source_clf = self.classifier(max_source_feature)

        # 计算最大损失源域的分类损失
        max_cls_loss = self.criterion(max_source_clf, max_source_label)

        return max_cls_loss
    

    def predict(self, x):
        self.eval()
        with torch.no_grad():
            feature = self.feature_extractor(x)
            preds = self.classifier.predict(feature)
        return preds
    
    def predict_prob(self, x):
        self.eval()
        with torch.no_grad():
            feature = self.feature_extractor(x)
            output = self.classifier(feature)
            logits = F.softmax(output, dim=1)
        return logits

    def get_parameters(self):
        params = [
            *self.feature_extractor.get_parameters(),
            *self.classifier.get_parameters(),
        ]
        # print(params)
        return params
    

    
