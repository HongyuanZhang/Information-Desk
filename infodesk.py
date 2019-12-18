import torch
from kse_predict import predict_all as kse_predict  # for predicting key sentence
from gan_predict import predict as gan_predict  # for generate natural answers


folder='logs/'  # data foler
# checkpoint file paths
kse_en_ckpt_path=folder+'kse_en.ckpt'
kse_end_ckpt_path=folder+'kse_end.ckpt'
kse_de_ckpt_path=folder+'kse_de.ckpt'
gan_en_ckpt_path=folder+'gan_en.ckpt'
gan_de_ckpt_path=folder+'gan_de.ckpt'
gan_dis_ckpt_path=folder+'gan_dis.ckpt'
# data set file paths
qa_path=folder+'output.csv'
fa_path=folder+'output_f_a.csv'

device = 'cuda' if torch.cuda.is_available() else 'cpu'  # computing device


if __name__ == '__main__':
    q = "What do you think of Emma Watson?"  # edit this string to the question you want to ask
    # predict key sentences from related web pages
    ks = kse_predict(q, kse_en_ckpt_path, kse_end_ckpt_path, kse_de_ckpt_path, device)
    # generate natural answers from those key sentences
    # 100: max number of words in generated answer
    ans = gan_predict(ks, 100, qa_path, fa_path, gan_en_ckpt_path, gan_de_ckpt_path, gan_dis_ckpt_path, device)
    print(ans)
