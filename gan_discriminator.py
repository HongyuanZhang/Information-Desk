# Citation: https://github.com/suragnair/seqGAN
import torch
import torch.nn as nn


# the Discriminator in the answer generator
class Discriminator(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, max_seq_len, gpu=False, dropout=0.2):
        # dropout: dropout rate; dropout is a mechanism for avoiding overfitting
        super(Discriminator, self).__init__()
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.max_seq_len = max_seq_len  # maximum sequence length in this model
        self.gpu = gpu  # whether use gpu to compute

        self.embeddings = nn.Embedding(vocab_size, embedding_dim)  # embedding layer
        self.gru = nn.GRU(embedding_dim, hidden_dim, num_layers=2, bidirectional=True, dropout=dropout)  # Gated Recurrent Unit
        self.gru2hidden = nn.Linear(2*2*hidden_dim, hidden_dim)  # a linear hidden layer
        self.dropout_linear = nn.Dropout(p=dropout)  # dropout layer
        self.hidden2out = nn.Linear(hidden_dim, 1)  # output layer

    def init_hidden(self, batch_size):
        '''
        initialize hidden state
        '''
        h = torch.zeros((2*2*1, batch_size, self.hidden_dim), requires_grad=True)
        return h

    def forward(self, input, hidden):
        '''
        hidden: hidden state
        forward input through the Discriminator
        '''
        # input dim                                                # batch_size x seq_len
        emb = self.embeddings(input)                               # batch_size x seq_len x embedding_dim
        emb = emb.permute(1, 0, 2)                                 # seq_len x batch_size x embedding_dim
        _, hidden = self.gru(emb, hidden)                          # 4 x batch_size x hidden_dim
        hidden = hidden.permute(1, 0, 2).contiguous()              # batch_size x 4 x hidden_dim
        out = self.gru2hidden(hidden.view(-1, 4*self.hidden_dim))  # batch_size x 4*hidden_dim
        out = torch.tanh(out)
        out = self.dropout_linear(out)
        out = self.hidden2out(out)                                 # batch_size x 1
        out = torch.sigmoid(out)
        return out


    def pairClassify(self, inp):
        """
        Classifies a batch of sequences.

        Inputs: inp
            - inp: batch_size x seq_len

        Returns: out
            - out: batch_size ([0,1] score)
        """
        h = torch.zeros(2,2,64, requires_grad=True)  # hidden state
        out = self.forward(inp, h)  # forward input through the network
        return out.view(-1)
