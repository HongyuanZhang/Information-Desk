import torch.nn as nn


def init_weights(m):
    # Initialize parameters (weights and biases)
    # for a neural network layer
    if type(m)==nn.Linear or type(m)==nn.Conv2d:
        nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)