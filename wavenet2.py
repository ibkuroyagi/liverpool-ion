import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import random
#from tqdm import tqdm
from tqdm.notebook import tqdm
import gc
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score
import warnings
class Attention(nn.Module):
    def __init__(self, feature_dim, step_dim, bias=True, **kwargs):
        super(Attention, self).__init__(**kwargs)
        
        self.supports_masking = True

        self.bias = bias
        self.feature_dim = feature_dim
        self.step_dim = step_dim
        self.features_dim = 0
        
        weight = torch.zeros(feature_dim, 1)
        nn.init.xavier_uniform_(weight)
        self.weight = nn.Parameter(weight)
        
        if bias:
            self.b = nn.Parameter(torch.zeros(step_dim))
        
    def forward(self, x, mask=None):
        feature_dim = self.feature_dim
        step_dim = self.step_dim

        eij = torch.mm(
            x.contiguous().view(-1, feature_dim), 
            self.weight
        ).view(-1, step_dim)
        
        if self.bias:
            eij = eij + self.b
            
        eij = torch.tanh(eij)
        a = torch.exp(eij)
        
        if mask is not None:
            a = a * mask

        a = a / torch.sum(a, 1, keepdim=True) + 1e-10

        weighted_input = x * torch.unsqueeze(a, -1)
        #return torch.sum(weighted_input, 1)
        return weighted_input




# from https://www.kaggle.com/hanjoonchoe/wavenet-lstm-pytorch-ignite-ver        
class Wave_Block(nn.Module):
    
    def __init__(self,in_channels, out_channels, dilation_rates):
        super(Wave_Block, self).__init__()
        self.num_rates = dilation_rates
        self.convs = nn.ModuleList()
        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        
        self.convs.append(nn.Conv1d(in_channels, out_channels, kernel_size=1))
        dilation_rates = [2**i for i in range(dilation_rates)]
        for dilation_rate in dilation_rates:
            self.filter_convs.append(nn.Conv1d(out_channels,out_channels,kernel_size=3,padding=dilation_rate,dilation=dilation_rate))
            self.gate_convs.append(nn.Conv1d(out_channels,out_channels,kernel_size=3,padding=dilation_rate,dilation=dilation_rate))
            self.convs.append(nn.Conv1d(out_channels,out_channels,kernel_size=1))
            
    def forward(self,x):
        x = self.convs[0](x)
        res = x
        for i in range(self.num_rates):
            x = F.tanh(self.filter_convs[i](x))*F.sigmoid(self.gate_convs[i](x))
            x = self.convs[i+1](x)
            x += res
        return x
  
  
class Classifier(nn.Module):
    def __init__(self, input_size=19, batch_size=5000):
        super().__init__()
        #input_size = 128
        self.batch_size = batch_size
        self.LSTM = nn.LSTM(input_size=128,hidden_size=64,num_layers=2,batch_first=True,bidirectional=True)
        self.attention = Attention(input_size,batch_size)
        #self.rnn = nn.RNN(input_size, 64, 2, batch_first=True, nonlinearity='relu')
        self.wave_block1 = Wave_Block(input_size,16,12)
        self.wave_block2 = Wave_Block(16,32,8)
        self.wave_block3 = Wave_Block(32,64,4)
        self.wave_block4 = Wave_Block(64, 128, 1)
        self.fc = nn.Linear(128, 11)
            
    def forward(self,x):
        x = x.permute(0, 2, 1)
        attention = self.attention(x)
        print(x.shape, attention.shape)
        x += attention
        x = x.permute(0, 2, 1)
        x = self.wave_block1(x)
        x = self.wave_block2(x)
        x = self.wave_block3(x)
       
        #x,_ = self.LSTM(x)
        x = self.wave_block4(x)
        x = x.permute(0, 2, 1)
        print(x.shape)
        x,_ = self.LSTM(x)
        print(x.shape)
        #x = self.conv1(x)
        #print(x.shape)
        #x = self.rnn(x)
        
        x = self.fc(x)
        return x


class EarlyStopping:
    def __init__(self, patience=5, delta=0, checkpoint_path='checkpoint.pt', is_maximize=True, device="cpu"):
        self.patience, self.delta, self.checkpoint_path = patience, delta, checkpoint_path
        self.counter, self.best_score = 0, None
        self.device = device
        self.is_maximize = is_maximize

    def load_best_weights(self, model):
        model.load_state_dict(torch.load(self.checkpoint_path, map_location=self.device))

    def __call__(self, score, model):
        if self.best_score is None or \
                (score > self.best_score + self.delta if self.is_maximize else score < self.best_score - self.delta):
            torch.save(model.to('cpu').state_dict(), self.checkpoint_path)
            model.to(self.device)
            self.best_score, self.counter = score, 0
            return 1
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return 2
        return 0
