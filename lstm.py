import os
from utility import *
import torch

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


class EncoderRNN(nn.Module):
    '''
    class for Encoder in the key sentence extractor
    encode the string (either question or document)
    '''
    def __init__(self, input_size, hidden_size, batch_size, device):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.device = device  # computing device
        self.embedding = nn.Embedding(input_size, hidden_size)  # embedding layer
        self.lstm = nn.LSTM(hidden_size, hidden_size, 1)  # long short-term memory network
        self.apply(init_weights)  # initialize network weights

    def forward(self, input, hidden, q_output=False):
        '''
        input - 16 x B
        hidden: hidden state
        '''
        if not q_output:
            embedded=self.embedding(input) # 16 x B x 256
        else:
            # the input is question encodings or document encodings
            # we do not need to embed
            embedded=input
        # run the lstm model
        output, hidden = self.lstm(embedded, hidden) # 16 x B x 256 
        return output, hidden

    def initHidden(self):
        # init the hidden layer to all zeros for the first round
        return (torch.zeros(1, self.batch_size, self.hidden_size, device=self.device),\
                torch.zeros(1, self.batch_size, self.hidden_size, device=self.device))


class Decoder(nn.Module):
    '''
    class for Decoder in the key sentence extractor
    extract the start index and end index given question encodings
    and document encodings
    '''
    def __init__(self, hidden_size, max_length):
        super(Decoder, self).__init__()
        # model structure: 4 convolutional networks and 5 linear layers
        self.conv1=nn.Sequential(
                nn.Conv1d(hidden_size,128,3),
                nn.BatchNorm1d(128),
                nn.MaxPool1d(kernel_size=3,stride=2,padding=0)
                )
        self.conv2=nn.Sequential(
                nn.Conv1d(128,64,3),
                nn.BatchNorm1d(64),
                nn.MaxPool1d(kernel_size=3,stride=2,padding=0)
                )
        self.conv3=nn.Sequential(
                nn.Conv1d(64,32,5),
                nn.BatchNorm1d(32),
                nn.MaxPool1d(kernel_size=5,stride=2,padding=0)
                )
        self.conv4=nn.Sequential(
                nn.Conv1d(32,16,5),
                nn.BatchNorm1d(16),
                nn.MaxPool1d(kernel_size=5,stride=2,padding=0)
                )

        self.fc1 = nn.Linear(160,64)
        self.fc2 = nn.Linear(64,32)
        self.fc3 = nn.Linear(32,16)
        self.fc4 = nn.Linear(16,4)
        self.fc5 = nn.Linear(4,2)
        # initialize network weights
        self.apply(init_weights)

    def forward(self, x):
        '''
        encoder_outputs - B x max_length x 256
        '''
        # pre-process input, x
        B=x.shape[0]
        x=x.transpose(1,2) # B x 256 x n

        x=self.conv1(x)
        x=self.conv2(x)
        x=self.conv3(x)
        x=self.conv4(x)
        x=x.view(B,-1)
        x=self.fc1(x)
        x=self.fc2(x)
        x=self.fc3(x)
        x=self.fc4(x)
        x=self.fc5(x)
        return x
