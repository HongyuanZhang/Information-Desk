from gan_train import *


def predict(input_string, max_length, qa_path, fa_path, en_ckpt_path, de_ckpt_path, dis_ckpt_path, device):
    '''
    input_string: an answer candidate
    The goal of this function is to modify input_string so that it looks natural and real-human-made
    For comments of this function, consult gan_train.py, since they are extremely similar
    '''
    real_questions, real_answers, fake_answers = load_stack_exchange_data(qa_path, fa_path)
    empty_ind = [i for i, x in enumerate(fake_answers) if x == '']
    nonempty_f_a = [x for i, x in enumerate(fake_answers) if i not in empty_ind]
    nonempty_r_a = [x for i, x in enumerate(real_answers) if i not in empty_ind]
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
    input_lang, output_lang, pairs = prepareData(nonempty_f_a, nonempty_r_a)
    MAX_LENGTH_DICTIONARY = max(input_lang.n_words, output_lang.n_words)
    discriminator = Discriminator(DIS_EMBEDDING_DIM, DIS_HIDDEN_DIM, MAX_LENGTH_DICTIONARY, MAX_LENGTH, gpu=CUDA)
    encoder = EncoderRNN(input_lang.n_words, GEN_HIDDEN_DIM)
    decoder = AttnDecoderRNN(GEN_HIDDEN_DIM, output_lang.n_words, MAX_LENGTH)

    if os.path.isfile(en_ckpt_path):
        encoder.load_state_dict(torch.load(en_ckpt_path))
    if os.path.isfile(de_ckpt_path):
        decoder.load_state_dict(torch.load(de_ckpt_path))
    if os.path.isfile(dis_ckpt_path):
        discriminator.load_state_dict(torch.load(dis_ckpt_path))

    # predict using networks
    with torch.no_grad():
        # convert input string into a tensor of word indices
        input_tensor = tensorFromSentence(input_lang, normalizeString(input_string))
        encoder_hidden = encoder.initHidden()
        input_length = input_tensor.size(0)
        gen_outputs = torch.zeros(max_length, device=device)
        encoder_outputs = torch.zeros(MAX_LENGTH, encoder.hidden_size, device=device)

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(
            input_tensor[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0, 0]

    decoder_input = torch.tensor([[SOS_token]], device=device)

    decoder_hidden = encoder_hidden

    for di in range(max_length):
        decoder_output, decoder_hidden, decoder_attention = decoder(
            decoder_input, decoder_hidden, encoder_outputs)
        topv, topi = decoder_output.topk(1)
        decoder_input = topi.squeeze().detach()
        gen_outputs[di] = topi.item()
        # if got end-of-string token, terminate the loop
        if decoder_input.item() == EOS_token:
            # generator's output is the output so far
            gen_outputs = gen_outputs[:di+1]
            break
    # convert predicted word indices into text
    sentence = []
    for i in range(gen_outputs.size(0)):
        sentence.append(output_lang.index2word[gen_outputs[i].item()])
    sentence = ' '.join(sentence)
    return sentence
