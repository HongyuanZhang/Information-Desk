import torch
import torch.nn as nn
import torch.nn.functional as F
from utility import *

from lstm import * 
from dataloader import *

dataset=GoogleNQ('simplified-nq-train.jsonl')
dataloader=DataLoader(dataset,batch_size=1,shuffle=True,num_workers=1)

SOS_token=char2int['SOS']
EOS_token=char2int['EOS']

lr=0.01
n_epoch=100

en_ckpt_path='en.ckpt'
de_ckpt_path='de.ckpt'
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

def train():
    encoder=EncoderRNN(input_size=len(charset), hidden_size=256, device=device)
    decoder=AttnDecoderRNN(hidden_size=256, output_size=len(charset), device=device)
    en_optimizer=torch.optim.Adam(encoder.parameters(),lr=lr,betas=(0.5,0.9))
    de_optimizer=torch.optim.Adam(decoder.parameters(),lr=lr,betas=(0.5,0.9))
    criterion = nn.NLLLoss()
    encoder.train()
    decoder.train()
    if os.path.isfile(en_ckpt_path):
        encoder.load_state_dict(torch.load(en_ckpt_path))
        print("Loaded encoder ckpt!")
    if os.path.isfile(de_ckpt_path):
        decoder.load_state_dict(torch.load(de_ckpt_path))
        print("Loaded decoder ckpt!")

    for e in range(n_epoch):
        step=0
        loss_sum=0
        for _,(qd,a) in enumerate(dataloader):
            predicted=list()
            en_optimizer.zero_grad()
            de_optimizer.zero_grad()
            qd,a=qd.to(device)[0],a.to(device)[0]
            encoder_hidden = encoder.initHidden()
            qd_len=qd.shape[0]
            a_len=a.shape[0]
            encoder_outputs = torch.zeros(decoder.max_length, encoder.hidden_size, device=device)
            loss = 0
            for ei in range(qd_len):
                encoder_output, encoder_hidden = encoder(qd[ei], encoder_hidden)
                if ei>=qd_len-decoder.max_length:
                    idx=ei-max(0,qd_len-decoder.max_length)
                    encoder_outputs[idx] = encoder_output[0,0]
            decoder_input = torch.tensor([[SOS_token]], device=device)
            decoder_hidden = encoder_hidden
            for di in range(a_len):
                decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs)
                topv, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze().detach()
                predicted.append(decoder_input.item())
                # TODO: implement custom loss later
                loss += criterion(decoder_output, a[di].view(1,))
                if decoder_input.item() == EOS_token:
                    break
            loss.backward()
            loss_sum+=loss
            en_optimizer.step()
            de_optimizer.step()
            #print(int2str(predicted))
            predicted=list()
            if step%100==99:
                torch.save(encoder.state_dict(), en_ckpt_path)
                torch.save(decoder.state_dict(), de_ckpt_path)
                print('Epoch [{}/{}] , Step {}, Loss: {:.4f}'
                        .format(e+1, epoch, step, loss_sum/100.0))
            step+=1
            loss_sum=0


if __name__ == '__main__':
    train()
