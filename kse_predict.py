import math
from lstm import * 
from dataloader import *


def predict(q, d, en_ckpt_path, end_ckpt_path, de_ckpt_path, device):
    '''
    predict the key sentence from one web page given one question
    q - a question string
    d - a context string (web page)
    '''
    # model-related parameters
    seq_len=1
    q_attention_len=16
    attention_len=240

    # count number of words (tokens)
    d_tok=d.split(" ")
    d_tok_len=len(d_tok)

    # convert string to int array
    q=str2int(q)
    d=str2int(d)
    q=torch.LongTensor(q).unsqueeze(0).to(device)
    d=torch.LongTensor(d).unsqueeze(0).to(device)

    # construct the models
    encoder=EncoderRNN(input_size=len(charset), hidden_size=256, batch_size=1, device=device).to(device)
    encoder_d=EncoderRNN(input_size=len(charset), hidden_size=256, batch_size=1, device=device).to(device)
    decoder=Decoder(hidden_size=256, max_length=attention_len+q_attention_len).to(device)
    encoder.eval()
    encoder_d.eval()
    decoder.eval()

    # load checkpoints
    if os.path.isfile(en_ckpt_path):
        encoder.load_state_dict(torch.load(en_ckpt_path))
    if os.path.isfile(end_ckpt_path):
        encoder_d.load_state_dict(torch.load(end_ckpt_path))
    if os.path.isfile(de_ckpt_path):
        decoder.load_state_dict(torch.load(de_ckpt_path))

    # run the models
    with torch.no_grad():
        q=q.transpose(0,1)
        d=d.transpose(0,1) # n x 1
        q_len=q.shape[0]
        d_len=d.shape[0]
        encoder_hidden = encoder.initHidden()
        q_round=int(math.ceil(q_len/float(seq_len)))
        d_round=int(math.ceil(d_len/float(seq_len)))
        start=0
        # to store the question encodings
        q_outputs=list()
        for ei in range(q_round):
            encoder_output, encoder_hidden = encoder(q[start:min(start+seq_len,q_len)], encoder_hidden)
            start+=seq_len
            q_outputs.append(encoder_output.transpose(0,1)) # B x 16 x 256
        q_outputs=torch.cat(q_outputs,dim=1) # B x 16*n x 256
        q_outputs_last=list()
        qo=q_outputs[0][:q_len] # since this is prediction code, there is only one entry in the batch
        qo=qo[-q_attention_len:] # 10 x 256
        q_outputs_last.append(qo.unsqueeze(0)) # 1 x 10 x 256
        q_outputs_last=torch.cat(q_outputs_last,dim=0) # B x 10 x 256
        q_outputs_last=q_outputs_last.transpose(0,1) # 10 x B x 256

        # to store the document encodings
        encoder_outputs=list()
        start=0
        encoder_hidden_d = encoder_d.initHidden()
        encoder_output, encoder_hidden_d = encoder_d(q_outputs_last, encoder_hidden_d, q_output=True)
        for ei in range(d_round):
            encoder_output, encoder_hidden_d = encoder_d(d[start:min(start+seq_len,d_len)], encoder_hidden_d)
            start+=seq_len
            encoder_outputs.append(encoder_output.transpose(0,1)) # B x 16 x 256
        encoder_outputs=torch.cat(encoder_outputs,dim=1) # B x 16*n x 256
        eo_truncated=list()
        eo=encoder_outputs[0][:d_len]
        eo=eo[-attention_len:]
        eo_truncated.append(eo.unsqueeze(0))
        encoder_outputs=torch.cat(eo_truncated,dim=0) # B x 100 x 256
        encoder_outputs_combined=torch.cat([q_outputs_last.transpose(0,1),encoder_outputs],dim=1) # B x 101 x 256

        #run through the encoder again
        encoder_outputs=list()
        start=0
        encoder_hidden_d = encoder_d.initHidden()
        encoder_output, encoder_hidden_d = encoder_d(encoder_outputs_combined.transpose(0,1), encoder_hidden_d, q_output=True)
        for ei in range(d_round):
            encoder_output, encoder_hidden_d = encoder_d(d[start:min(start+seq_len,d_len)], encoder_hidden_d)
            start+=seq_len
            encoder_outputs.append(encoder_output.transpose(0,1)) # B x 16 x 256
        encoder_outputs=torch.cat(encoder_outputs,dim=1) # B x 16*n x 256
        eo_truncated=list()
        eo=encoder_outputs[0][:d_len]
        eo=eo[-attention_len:]
        eo_truncated.append(eo.unsqueeze(0))
        encoder_outputs=torch.cat(eo_truncated,dim=0) # B x 100 x 256
        encoder_outputs_combined=torch.cat([q_outputs_last.transpose(0,1),encoder_outputs],dim=1) # B x 101 x 256
        # get the start and end indices
        decoder_output=decoder(encoder_outputs_combined) # B x 2
        decoder_output+=0.5
        decoder_output=decoder_output.clamp(0,1)
        ss=(decoder_output[0][0]*d_tok_len).round().clamp(0,d_tok_len-2).long()
        ee=(decoder_output[0][1]*d_tok_len).round().clamp(ss+1,d_tok_len-1).long()

        # extract the key sentence
        predicted=d_tok[ss:ee]
        predicted=" ".join(predicted)
        return predicted


def predict_all(q, en_ckpt_path, end_ckpt_path, de_ckpt_path, device):
    # run the predict function on all the web pages fetched by the web loader
    ds=get_webpages(q)
    ans=list()
    for d in ds:
        # skip web pages that are too short
        if len(d)<1000:
            continue
        # append the current key sentence
        ans.append(predict(q, d, en_ckpt_path, end_ckpt_path, de_ckpt_path, device))
    max_len=0
    max_idx=0
    # choose the longest key sentence
    for i in range(len(ans)):
        if len(ans[i])>max_len:
            max_len=len(ans[i])
            max_idx=i
    return ans[max_idx]
