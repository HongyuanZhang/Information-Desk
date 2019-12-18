import random
import torch.optim as optim
import csv
import os
from gan_discriminator import *
from gan_generator import *
from dataloader import *


CUDA = False  # computing device
VOCAB_SIZE = len(charset)
teacher_forcing_ratio = 0.5
DIS_EMBEDDING_DIM = 64  # discriminator embedding dimension
DIS_HIDDEN_DIM = 64  # discriminator hidden dimension
GEN_HIDDEN_DIM = 256  # generator hidden dimension
folder = 'logs/'  # data folder path
stack_exchange_data_path = folder+'output.csv'  # 3515 Q&A's from Stack Exchange
fake_answer_path = folder+'output_f_a.csv'  # answer candidates corresponding to the 3515 Q&A's
en_ckpt_path = folder+'gan_en.ckpt'  # encoder checkpoint file path
de_ckpt_path = folder+'gan_de.ckpt'  # decoder checkpoint file path
dis_ckpt_path = folder+'gan_dis.ckpt'  # discriminator checkpoint file path


def load_stack_exchange_data(qa_path, fa_path):
    '''
    qa_path: path of the file that contains Q&A's from Stack Exchange
    fa_path: path of the file the contains corresponding answer candidates
    '''
    real_questions = []
    real_answers = []
    fake_answers = []
    # read Stack Exchange data set
    with open(qa_path, 'r', encoding='windows-1252') as rf:
        reader = csv.reader(rf, delimiter=',')
        for row in reader:
            # require that the real answer is not empty and contains common characters
            if row[2] != '' and row[2].isprintable():
                real_questions.append(row[0])  # add Q
                real_answers.append(row[2])  # add A
    # read Answer Candidate data set
    with open(fa_path, newline='', encoding='utf8') as f:
        reader = csv.reader(f)
        fake_answers = next(reader)
    return real_questions, real_answers, fake_answers


def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer,
          max_length, discriminator, discriminator_optimizer, output_lang):
    '''
    Given input sequence and target output, train generator (encoder & decoder)
    and discriminator
    '''
    encoder_hidden = encoder.initHidden()
    # zero out the gradients in optimizers
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    discriminator_optimizer.zero_grad()
    # loss function: binary cross-entropy
    loss_fn = nn.BCELoss()
    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)
    # initialize losses for generator and discriminator
    g_loss = torch.zeros(1)
    d_loss = torch.zeros(1)
    # allow for fine grained exclusion of subgraphs from gradient computation
    g_loss.requires_grad = True
    d_loss.requires_grad = True
    # placeholder: output from generator
    gen_outputs = torch.zeros(target_length, device=device)
    # placeholder: output from encoder
    encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

    # forward through encoder
    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(
            input_tensor[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0, 0]
    # initial decoder input: end-of-string token
    decoder_input = torch.tensor([[SOS_token]], device=device)
    # initial decoder hidden state: encoder hidden state
    decoder_hidden = encoder_hidden
    input_tensor = input_tensor.float()
    # convert input tensor from indices to list of words
    input_words = [input_lang.index2word[i.item()] for i in input_tensor]
    # Teacher forcing: Feed the target as the next input
    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
    if use_teacher_forcing:
        # forward through decoder
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            decoder_input = target_tensor[di]  # Teacher forcing
            gen_outputs[di] = target_tensor[di]

    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()  # most probable output
            gen_outputs[di] = topi
            # if the output word is not in input sequence of words, penalize
            if not(output_lang.index2word[gen_outputs[di].item()] in input_words):
                g_loss += 1
            if decoder_input.item() == EOS_token:
                break
    # normalize loss
    g_loss /= target_length
    # dicriminator's judgement on generator's output and real answer
    dis_decision = discriminator.pairClassify(torch.stack((gen_outputs.float(), torch.squeeze(target_tensor).float())).long())
    # discriminator's target judgment
    dis_target = torch.tensor([0., 1.])
    # penalize discriminator for wrong judgment
    d_loss += loss_fn(dis_decision, dis_target)
    # back propagate gradients and update network parameters
    d_loss.backward()
    discriminator_optimizer.step()
    # penalize the generator if it fails to fool the discriminator
    g_loss += loss_fn(dis_decision[0].reshape(1), torch.ones(1))
    # back propagate gradients and update network parameters
    g_loss.backward()
    encoder_optimizer.step()
    decoder_optimizer.step()
    print("Discriminator Loss: ", d_loss)
    print("Generator Loss: ", g_loss)
    # convert output tensor of word indices to text
    sentence = []
    for i in range(target_length):
        sentence.append(output_lang.index2word[gen_outputs[i].item()])
    sentence = ' '.join(sentence)
    print("Generated Answer: ", sentence)


if __name__ == '__main__':
    # load data
    real_questions, real_answers, fake_answers = load_stack_exchange_data(stack_exchange_data_path, fake_answer_path)
    # filter out empty data
    empty_ind = [i for i,x in enumerate(fake_answers) if x == '']
    nonempty_f_a = [x for i, x in enumerate(fake_answers) if i not in empty_ind]
    nonempty_r_a = [x for i, x in enumerate(real_answers) if i not in empty_ind]
    # filter out overly long data
    longest_str = max(nonempty_r_a, key=len)
    longest_ind = nonempty_r_a.index(longest_str)
    del nonempty_r_a[longest_ind]
    del nonempty_f_a[longest_ind]
    second_longest_str = max(nonempty_r_a, key=len)
    second_longest_ind = nonempty_r_a.index(second_longest_str)
    del nonempty_r_a[second_longest_ind]
    del nonempty_f_a[second_longest_ind]

    MAX_LENGTH_FAKE_ANS = max([len(a.split(' ')) for a in nonempty_f_a])
    MAX_LENGTH_REAL_ANS = max([len(a.split(' ')) for a in nonempty_r_a])
    MAX_LENGTH = max(MAX_LENGTH_FAKE_ANS, MAX_LENGTH_REAL_ANS)
    # read data into lang objects
    input_lang, output_lang, pairs = prepareData(nonempty_f_a, nonempty_r_a)
    MAX_LENGTH_DICTIONARY = max(input_lang.n_words, output_lang.n_words)
    # create networks
    discriminator = Discriminator(DIS_EMBEDDING_DIM, DIS_HIDDEN_DIM, MAX_LENGTH_DICTIONARY, MAX_LENGTH, gpu=CUDA)
    encoder = EncoderRNN(input_lang.n_words, GEN_HIDDEN_DIM)
    decoder = AttnDecoderRNN(GEN_HIDDEN_DIM, output_lang.n_words, MAX_LENGTH)
    # network optimizers
    encoder_optimizer = optim.SGD(encoder.parameters(), lr=0.1)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=0.1)
    discriminator_optimizer = optim.Adagrad(discriminator.parameters())
    # load checkpoint files, if exist
    if os.path.isfile(en_ckpt_path):
        encoder.load_state_dict(torch.load(en_ckpt_path))
    if os.path.isfile(de_ckpt_path):
        decoder.load_state_dict(torch.load(de_ckpt_path))
    if os.path.isfile(dis_ckpt_path):
        discriminator.load_state_dict(torch.load(dis_ckpt_path))
    # train on data
    with torch.no_grad():
        # train pair by pair
        for iter in range(len(nonempty_f_a)):
            input_tensor = tensorFromSentence(input_lang, normalizeString(pairs[iter][0]))
            target_tensor = tensorFromSentence(output_lang, normalizeString(pairs[iter][1]))
            train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, MAX_LENGTH,
              discriminator, discriminator_optimizer, output_lang)
            # store checkpoints at times
            if iter % 100 == 0:
                torch.save(encoder.state_dict(), en_ckpt_path)
                torch.save(decoder.state_dict(), de_ckpt_path)
                torch.save(discriminator.state_dict(), dis_ckpt_path)
