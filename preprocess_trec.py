import argparse, time, torch, json
import opts, util
import pandas as pd
from tqdm import tqdm
from transformers import *
import pdb

import torch.nn as nn
cos = nn.CosineSimilarity(dim=1)

parser = argparse.ArgumentParser(
    description='preprocess_trec.py')
opts.preprocess_opts(parser)
opt = parser.parse_args()

b_time = time.time()

with open(opt.input, 'r') as input_f, open(opt.output, 'w') as output_f:
    # data preparation
    test_dataset = json.load(input_f)

    d_idx = 0
    for d in test_dataset:

        u = d['dialogue']
        pa = [p for _, p in d['pa'].items() if not pd.isnull(p) and len(p) > 0]
        pb = [p for _, p in d['pb'].items() if not pd.isnull(p) and len(p) > 0]

        for u_idx in range(len(u)):
            links = u[u_idx]['A'][1]
            for l in links:
                if l > 0:
                    output_f.write("d{} 0 p{}_u{} 1\n".format(d_idx, l,u_idx))
        d_idx += 1

        for u_idx in range(len(u)):
            links = u[u_idx]['B'][1]
            for l in links:
                if l > 0:
                    output_f.write("d{} 0 p{}_u{} 1\n".format(d_idx, l, u_idx))
        d_idx += 1


e_time = time.time()

print('time: {}'.format(e_time - b_time))
