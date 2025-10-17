# -*- encoding: utf-8 -*-
'''
file       :Base.py
Date       :2024/07/22 11:57:17
Author     :qwangwl
'''

# 和Model_PR_PL相比，将分类器进行了修改。

import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from loss_funcs import TransferLoss

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


class Base(nn.Module):
    def __init__(self, 
                 input_dim: int= 310, 
                 num_of_class: int=3, 
                 max_iter: int=1000, 
                 transfer_loss_type: str="dann", 
                 **kwargs):
        super(Base, self).__init__()

        self.feature_extractor = FeatureExtractor(input_dim=input_dim)
        self.classifier = LabelClassifier(input_dim=64, num_of_class=num_of_class)
        
        
        self.criterion = nn.CrossEntropyLoss()
    def forward(self, source, source_label):
        source_feature = self.feature_extractor(source)
        source_output = self.classifier(source_feature)
        cls_loss = self.criterion(source_output, source_label)

        return cls_loss
    

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
    

    
