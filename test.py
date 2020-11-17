
import torch

import opts, util
import numpy as np
from tqdm import tqdm
import argparse, sys

import nn_model

parser = argparse.ArgumentParser(
    description='train.py')
opts.test_opts(parser)
opt = parser.parse_args()

np.random.seed(opt.seed)

def test():
    test_dataset = torch.load(opt.test_dataset)
    if opt.model is not None:
        model = torch.load(opt.model)
    else:
        model = None

    if opt.cuda:
        test_utterance = [u.cuda() for u in test_dataset['utterance']]
        test_persona = [u.cuda() for u in test_dataset['persona']]

        if model:
            model = model.cuda()
    else:
        test_utterance = test_dataset['utterance']
        test_persona = test_dataset['persona']

    assert len(test_persona) == len(test_utterance)

    with open(opt.test_result, 'w') as result_f:
        for i in range(len(test_utterance)):
            if opt.method == 'bert' or opt.method == 'rand':
                utt_rep, per_rep = test_utterance[i], test_persona[i]
            else:
                utt_rep, per_rep = model.linear(test_utterance[i]), model.linear(test_persona[i])
            sim = nn_model.pairwise_cosine(utt_rep, per_rep).data.cpu().numpy()
            u_num, p_num = sim.shape
            for u_idx in range(u_num):
                for p_idx in range(p_num):
                    if opt.method == 'rand':
                        result_f.write("d{} Q0 p{}_u{} 0 {} STANDARD\n".format(i, p_idx+1, u_idx, np.random.rand()))
                    else:
                        result_f.write("d{} Q0 p{}_u{} 0 {} STANDARD\n".format(i, p_idx+1, u_idx,  sim[u_idx, p_idx]))


if __name__ == '__main__':
    test()