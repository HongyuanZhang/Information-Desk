from lstm import *
from dataloader import *
import torch.nn.functional as F


SOS_token=char2int['SOS']  # SOS: start-of-string
EOS_token=char2int['EOS']  # EOS: end-of-string


# training parameters
lr=0.00003  # learning rate
n_epoch=1
seq_len=1
q_attention_len=16
attention_len=240
batch_size=8

# checkpoint file paths
folder='logs/'
en_ckpt_path=folder+'kse_en.ckpt'
end_ckpt_path=folder+'kse_end.ckpt'
de_ckpt_path=folder+'kse_de.ckpt'
device = 'cuda' if torch.cuda.is_available() else 'cpu'  # computing device


def train(buc_lim_idx):
    '''
    train the key sentence extractor using entries in the bucket of size buc_lim[buc_lim_idx]
    '''
    # construct dataset (defined in dataloader.py) and dataloader
    dataset=GoogleNQ(buc_lim[buc_lim_idx],q_lim,a_lim)
    dataloader=DataLoader(dataset,batch_size=batch_size,shuffle=True,num_workers=1)

    # construct models
    encoder=EncoderRNN(input_size=len(charset), hidden_size=256, batch_size=batch_size, device=device).to(device)
    encoder_d=EncoderRNN(input_size=len(charset), hidden_size=256, batch_size=batch_size, device=device).to(device)
    decoder=Decoder(hidden_size=256, max_length=attention_len+q_attention_len).to(device)

    # model optimizers
    en_optimizer=torch.optim.Adam(encoder.parameters(),lr=lr,betas=(0.5,0.9))
    end_optimizer=torch.optim.Adam(encoder_d.parameters(),lr=lr,betas=(0.5,0.9))
    de_optimizer=torch.optim.Adam(decoder.parameters(),lr=lr,betas=(0.5,0.9))

    # loss
    encoder.train()
    encoder_d.train()
    decoder.train()

    # load checkpoints
    if os.path.isfile(en_ckpt_path):
        encoder.load_state_dict(torch.load(en_ckpt_path))
        print("Loaded encoder ckpt!")
    if os.path.isfile(end_ckpt_path):
        encoder_d.load_state_dict(torch.load(end_ckpt_path))
        print("Loaded encoder_d ckpt!")
    if os.path.isfile(de_ckpt_path):
        decoder.load_state_dict(torch.load(de_ckpt_path))
        print("Loaded decoder ckpt!")

    # train loop
    for e in range(n_epoch):
        step=0
        loss_sum=0
        for _,(q,d,q_len,d_len,ss,ee) in enumerate(dataloader):
            # since lstm models only accept fixed batch size
            if q.shape[0] != batch_size:
                continue
            # zero out gradients in optimizers
            en_optimizer.zero_grad()
            end_optimizer.zero_grad()
            de_optimizer.zero_grad()

            # pre-process inputs
            q,d,ss,ee=q.to(device),d.to(device),ss.to(device),ee.to(device) # B x n
            se=torch.cat([ss,ee],dim=1) # B x 2
            q=q.transpose(0,1)
            d=d.transpose(0,1) # n x B
            q_len,d_len=q_len.to(device),d_len.to(device) # B x 1
            q_max,d_max=q_len.max(),d_len.max()
            # initialize encoder hidden state
            encoder_hidden = encoder.initHidden()
            # initialize loss value
            loss = 0

            # how many rounds should question encoder and document encoder run
            q_round=(q_max/float(seq_len)).ceil().long()
            d_round=(d_max/float(seq_len)).ceil().long()
            start=0
            q_outputs=list()
            
            # run the question encoder
            for ei in range(q_round):
                encoder_output, encoder_hidden = encoder(q[start:min(start+seq_len,q_max)], encoder_hidden)
                start+=seq_len
                q_outputs.append(encoder_output.transpose(0,1)) # B x 16 x 256
            q_outputs=torch.cat(q_outputs,dim=1) # B x 16*n x 256
            q_outputs_last=list()

            # extract the last few question encodings 
            # (make sure we don't include those blank outputs due to uneven lengths of the entries)
            for b in range(batch_size):
                qo=q_outputs[b][:q_len[b]]
                qo=qo[-q_attention_len:] # 10 x 256
                q_outputs_last.append(qo.unsqueeze(0)) # 1 x 10 x 256
            q_outputs_last=torch.cat(q_outputs_last,dim=0) # B x 10 x 256
            q_outputs_last=q_outputs_last.transpose(0,1) # 10 x B x 256

            encoder_outputs=list()
            start=0
            encoder_hidden_d = encoder_d.initHidden()
            encoder_output, encoder_hidden_d = encoder_d(q_outputs_last, encoder_hidden_d, q_output=True)
            # run the document encoder
            for ei in range(d_round):
                encoder_output, encoder_hidden_d = encoder_d(d[start:min(start+seq_len,d_max)], encoder_hidden_d)
                start+=seq_len
                encoder_outputs.append(encoder_output.transpose(0,1)) # B x 16 x 256
            encoder_outputs=torch.cat(encoder_outputs,dim=1) # B x 16*n x 256
            eo_truncated=list()

            # extract the last few document encodings
            for b in range(batch_size):
                eo=encoder_outputs[b][:d_len[b]]
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
                encoder_output, encoder_hidden_d = encoder_d(d[start:min(start+seq_len,d_max)], encoder_hidden_d)
                start+=seq_len
                encoder_outputs.append(encoder_output.transpose(0,1)) # B x 16 x 256
            encoder_outputs=torch.cat(encoder_outputs,dim=1) # B x 16*n x 256
            eo_truncated=list()
            for b in range(batch_size):
                eo=encoder_outputs[b][:d_len[b]]
                eo=eo[-attention_len:]
                eo_truncated.append(eo.unsqueeze(0))
            encoder_outputs=torch.cat(eo_truncated,dim=0) # B x 100 x 256
            encoder_outputs_combined=torch.cat([q_outputs_last.transpose(0,1),encoder_outputs],dim=1) # B x 101 x 256
            decoder_output=decoder(encoder_outputs_combined) # B x 2

            # calculate loss
            loss=F.smooth_l1_loss(decoder_output,se-0.5)
            # calculate gradients
            loss.backward()
            loss_sum+=loss.item()
            # back propagate gradients
            en_optimizer.step()
            end_optimizer.step()
            de_optimizer.step()
            # store checkpoints at times
            if step%2==1:
                torch.save(encoder.state_dict(), en_ckpt_path)
                torch.save(encoder_d.state_dict(), end_ckpt_path)
                torch.save(decoder.state_dict(), de_ckpt_path)
                # print state
                print('Epoch [{}/{}] , Step {}, Loss: {:.4f}'
                        .format(e+1, n_epoch, step, loss_sum/100.0))
                # re-set sum
                loss_sum=0
            step+=1
            # delete data
            del encoder_outputs
            del encoder_hidden
            del encoder_output
            del q_outputs
            del q_outputs_last
            del encoder_hidden_d
            del decoder_output


if __name__ == '__main__':
    # train n_epoch for each buc_lim
    for bli in range(len(buc_lim)):
        train(bli)
