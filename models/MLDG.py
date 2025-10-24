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

import torch.autograd as autograd
import torch.nn.functional as F
from torch.autograd import Variable


def linear(inputs, weight, bias, meta_step_size=0.001, meta_loss=None, stop_gradient=False):
    if meta_loss is not None:

        if not stop_gradient:
            grad_weight = autograd.grad(meta_loss, weight, create_graph=True)[0]

            if bias is not None:
                grad_bias = autograd.grad(meta_loss, bias, create_graph=True)[0]
                bias_adapt = bias - grad_bias * meta_step_size
            else:
                bias_adapt = bias

        else:
            grad_weight = Variable(autograd.grad(meta_loss, weight, create_graph=True)[0].data, requires_grad=False)

            if bias is not None:
                grad_bias = Variable(autograd.grad(meta_loss, bias, create_graph=True)[0].data, requires_grad=False)
                bias_adapt = bias - grad_bias * meta_step_size
            else:
                bias_adapt = bias

        return F.linear(inputs,
                        weight - grad_weight * meta_step_size,
                        bias_adapt)
    else:
        
        return F.linear(inputs, weight, bias)

    def forward(self, x, meta_loss=None, meta_step_size=None, stop_gradient=False):

        x = linear(inputs=x,
                   weight=self.fc1.weight,
                   bias=self.fc1.bias,
                   meta_loss=meta_loss,
                   meta_step_size=meta_step_size,
                   stop_gradient=stop_gradient)

        x = F.relu(x, inplace=True)

        x = linear(inputs=x,
                   weight=self.fc2.weight,
                   bias=self.fc2.bias,
                   meta_loss=meta_loss,
                   meta_step_size=meta_step_size,
                   stop_gradient=stop_gradient)

        end_points = {'Predictions': F.softmax(input=x, dim=-1)}

        return x, end_points
    
class FeatureExtractor(nn.Module):
    def __init__(self, input_dim=310, hidden_1=64, hidden_2=64):
        super(FeatureExtractor, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_1)
        self.fc2 = nn.Linear(hidden_1, hidden_2)

    def forward(self, x, meta_loss=None, meta_step_size=None, stop_gradient=False):
        x = linear(inputs=x,
                   weight=self.fc1.weight,
                   bias=self.fc1.bias,
                   meta_loss=meta_loss,
                   meta_step_size=meta_step_size if meta_step_size is not None else 0.001,
                   stop_gradient=stop_gradient)
        x = F.relu(x)
        x = linear(inputs=x,
                   weight=self.fc2.weight,
                   bias=self.fc2.bias,
                   meta_loss=meta_loss,
                   meta_step_size=meta_step_size if meta_step_size is not None else 0.001,
                   stop_gradient=stop_gradient)
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


class MLDG(nn.Module):
    def __init__(self, 
                 input_dim: int= 310, 
                 num_of_class: int=3, 
                 max_iter: int=1000, 
                 transfer_loss_type: str="dann", 
                 **kwargs):
        super(MLDG, self).__init__()

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
    

    
