import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from utility import *

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

attention_len=100

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, device, n_layers=2):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.device = device
        self.n_layers=n_layers

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, n_layers)
        self.apply(init_weights)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output, hidden = self.lstm(embedded, hidden)
        return output, hidden

    def initHidden(self):
        return (torch.zeros(self.n_layers, 1, self.hidden_size, device=self.device),\
                torch.zeros(self.n_layers, 1, self.hidden_size, device=self.device))

class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, device, n_layers=2, dropout_p=0.1, max_length=attention_len):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.device = device 
        self.n_layers=n_layers
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 3, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.lstm = nn.LSTM(self.hidden_size, self.hidden_size, n_layers)
        self.out = nn.Linear(self.hidden_size, self.output_size)
        self.apply(init_weights)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)

        attn_weights = F.softmax(self.attn(torch.cat((embedded[0], hidden[0][0], hidden[0][1]), 1)), dim=1) # 1 x att_len
        attn_applied = torch.bmm(attn_weights.unsqueeze(0), encoder_outputs.unsqueeze(0)) # 1 x 1 x hs

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0) # 1 x 1 x hs

        output = F.relu(output)
        output, hidden = self.lstm(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1) # 1 x hs
        return output, hidden, attn_weights

    def initHidden(self):
        return (torch.zeros(self.n_layers, 1, self.hidden_size, device=self.device),\
                torch.zeros(self.n_layers, 1, self.hidden_size, device=self.device))
