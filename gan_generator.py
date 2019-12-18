from __future__ import unicode_literals, print_function, division
import unicodedata
import re
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # computing device
SOS_token = 0  # SOS: start-of-string
EOS_token = 1  # EOS: end-of-string


# class for storing words and frequencies of target data set (real Stack Exchange answers)
# and input data set (answer candidates)
class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}  # a dictionary whose keys are words and values are indices
        self.word2count = {}  # a dictionary whose keys are words and values are counts
        self.index2word = {0: "SOS", 1: "EOS"}  # a dictionary whose keys are indices and values are words
        self.n_words = 2  # Count SOS and EOS

    def addSentence(self, sentence):
        '''
        add words in a sentence to this lang object
        '''
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        '''
        add a word to this lang object by creating/updating entries in the three
        dictionaries: word2index, word2count, index2word, as well as number of words
        in this lang object
        '''
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1


# Turn a Unicode string to plain ASCII, thanks to
# https://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )


# Lowercase, trim, and remove non-letter characters
def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    whitelist = set('abcdefghijklmnopqrstuvwxyz ABCDEFGHIJKLMNOPQRSTUVWXYZ')
    s = ''.join(filter(whitelist.__contains__, s))
    return s


def readLangs(fake_answers, real_answers):
    '''
    fake_answers: input data set, in this case answer candidate data
    real_answers: target data set, in this case real Stack Exchange answers
    make pairs of fake_answers and real_answers and lang objects
    '''
    # Split every line into pairs and normalize
    pairs = [[normalizeString(i), normalizeString(j)] for i,j in zip(fake_answers, real_answers)]
    # make Lang instances
    input_lang = Lang("fake")
    output_lang = Lang("real")
    return input_lang, output_lang, pairs


def prepareData(fake_answers, real_answers):
    '''
    fake_answers, real_answers: see comments above
    read data sets into lang objects
    '''
    input_lang, output_lang, pairs = readLangs(fake_answers, real_answers)
    # add pair by pair
    for pair in pairs:
        input_lang.addSentence(pair[0])  # add in fake answer
        output_lang.addSentence(pair[1])  # add in corresponding real answer
    return input_lang, output_lang, pairs


# class for encoder in the generator
class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)  # embedding layer
        self.gru = nn.GRU(hidden_size, hidden_size)  # gated recurrent unit

    def forward(self, input, hidden):
        '''
        hidden: hidden state
        forward input through the encoder
        '''
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self):
        '''
        initialize hidden state
        '''
        return torch.zeros(1, 1, self.hidden_size, device=device)


# class for decoder in the generator
class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(output_size, hidden_size)  # embedding layer
        self.gru = nn.GRU(hidden_size, hidden_size)  # gated recurrent unit
        self.out = nn.Linear(hidden_size, output_size)  # output layer
        self.softmax = nn.LogSoftmax(dim=1)  # softmax function normalizes input into a probability distribution

    def forward(self, input, hidden):
        '''
        hidden: hidden state
        forward input through the decoder
        '''
        output = self.embedding(input).view(1, 1, -1)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def initHidden(self):
        '''
        initialize hidden state
        '''
        return torch.zeros(1, 1, self.hidden_size, device=device)


# Decoder class with attention mechanism
class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, max_length, dropout_p=0.1):
        '''
        max_length: max sequence length in this model
        dropout_p: dropout rate; dropout is a mechanism for avoiding overfitting
        '''
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length
        self.embedding = nn.Embedding(self.output_size, self.hidden_size)  # embedding layer
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)  # attention layers
        self.dropout = nn.Dropout(self.dropout_p)  # dropout layer
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)  # gated recurrent unit
        self.out = nn.Linear(self.hidden_size, self.output_size)  # output layer

    def forward(self, input, hidden, encoder_outputs):
        '''
        hidden: hidden state
        forward input through the attention decoder
        '''
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)
        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))
        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = F.log_softmax(self.out(output[0]), dim=1)  # normalizes into probabilities
        return output, hidden, attn_weights

    def initHidden(self):
        '''
        initialize hidden state
        '''
        return torch.zeros(1, 1, self.hidden_size, device=device)


def indexesFromSentence(lang, sentence):
    '''
    convert a string into a list of indices of words in lang
    '''
    return [lang.word2index[word] for word in sentence.split(' ') if word in lang.word2index.keys()]


def tensorFromSentence(lang, sentence):
    '''
    convert a string into a tensor of indices of words in lang
    with EOS token appended
    '''
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)


def tensorsFromPair(pair, input_lang, output_lang):
    '''
    given a pair of strings, return a pair of tensors
    the first tensor is pair[0] converted into indices in input_lang
    the second tensor is pair[1] converted into indices in output_lang
    '''
    input_tensor = tensorFromSentence(input_lang, pair[0])
    target_tensor = tensorFromSentence(output_lang, pair[1])
    return (input_tensor, target_tensor)
