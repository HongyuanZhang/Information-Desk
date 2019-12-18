from lstm import *
from dataloader import *
'''
NOTE: The code in this file is extremely similar to that in kse_train.py.
See kse_train.py for detailed comments.
'''
SOS_token=char2int['SOS']
EOS_token=char2int['EOS']

n_epoch=1
seq_len=1
q_attention_len=16
attention_len=240
batch_size=8

folder='logs/'
en_ckpt_path=folder+'kse_en.ckpt'
end_ckpt_path=folder+'kse_end.ckpt'
de_ckpt_path=folder+'kse_de.ckpt'
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def test(buc_lim_idx):
    '''
    Evaluation code for key sentence extractor
    We evaluate for each bucket size (divided according to document length)
    The score is the average intersection-over-union percentage of predicted
    key sentence and ground-truth key sentence
    '''
    dataset=GoogleNQ(buc_lim[buc_lim_idx],q_lim,a_lim,test=True)
    dataloader=DataLoader(dataset,batch_size=batch_size,shuffle=True,num_workers=1)

    encoder=EncoderRNN(input_size=len(charset), hidden_size=256, batch_size=batch_size, device=device).to(device)
    encoder_d=EncoderRNN(input_size=len(charset), hidden_size=256, batch_size=batch_size, device=device).to(device)
    decoder=Decoder(hidden_size=256, max_length=attention_len+q_attention_len).to(device)
    encoder.eval()
    encoder_d.eval()
    decoder.eval()

    if os.path.isfile(en_ckpt_path):
        encoder.load_state_dict(torch.load(en_ckpt_path))
    if os.path.isfile(end_ckpt_path):
        encoder_d.load_state_dict(torch.load(end_ckpt_path))
    if os.path.isfile(de_ckpt_path):
        decoder.load_state_dict(torch.load(de_ckpt_path))

    iou_sum=0  # intersection-over-union sum
    count=0
    with torch.no_grad():
        for _,(q,d,q_len,d_len,ss,ee) in enumerate(dataloader):
            if q.shape[0] != batch_size:
                continue
            count+=q.shape[0]
            q,d,ss,ee=q.to(device),d.to(device),ss.to(device),ee.to(device) # B x n
            se=torch.cat([ss,ee],dim=1) # B x 2
            q=q.transpose(0,1)
            d=d.transpose(0,1) # n x B
            q_len,d_len=q_len.to(device),d_len.to(device) # B x 1
            q_max,d_max=q_len.max(),d_len.max()
            encoder_hidden = encoder.initHidden()
            loss = 0
            q_round=(q_max/float(seq_len)).ceil().long()
            d_round=(d_max/float(seq_len)).ceil().long()
            start=0
            q_outputs=list()
            for ei in range(q_round):
                encoder_output, encoder_hidden = encoder(q[start:min(start+seq_len,q_max)], encoder_hidden)
                start+=seq_len
                q_outputs.append(encoder_output.transpose(0,1)) # B x 16 x 256
            q_outputs=torch.cat(q_outputs,dim=1) # B x 16*n x 256
            q_outputs_last=list()
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

            # evaluate using intersection-over-union
            for b in range(batch_size):
                so=decoder_output[b][0]
                eo=decoder_output[b][1]
                sg=se[b][0]-0.5
                eg=se[b][1]-0.5
                union=max(eo,eg)-min(so,sg)
                inter=max(0,min(eo,eg)-max(so,sg))
                iou=inter/union
                iou_sum+=iou

            del encoder_outputs
            del encoder_hidden
            del encoder_output
            del q_outputs
            del q_outputs_last
            del encoder_hidden_d
            del decoder_output
            print("#",end="",flush=True)
    print()
    print("iou_sum:",iou_sum)
    print("count:",count)
    print("iou_avg:",iou_sum/count)


if __name__ == '__main__':
    # run tests
    for bli in range(len(buc_lim)):
        test(bli)
        print('============')
