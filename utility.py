import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def init_weights(m):
    # Initialize parameters
    if type(m)==nn.Linear or type(m)==nn.Conv2d:
        nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

def freeze_params(m):
    for param in m.parameters():
        param.requires_grad = False

def unfreeze_params(m):
    for param in m.parameters():
        param.requires_grad = True
