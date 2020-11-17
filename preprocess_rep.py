import argparse, time, torch, json
import opts, util
from tqdm import tqdm
from transformers import *
import pdb

import torch.nn as nn
cos = nn.CosineSimilarity(dim=1)

parser = argparse.ArgumentParser(
    description='preprocess_rep.py')
opts.preprocess_opts(parser)
opt = parser.parse_args()



if opt.lm == 'bert':
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')


with open(opt.input, 'r') as f:
    # data preparation
    train_dataset = json.load(f)
    samples_text, utterances, personas = util.extract_all_samples(train_dataset)

    print('dialogues[{}], utterances[{}], personas[{}]'.format(
        len(samples_text) / 2, len(utterances), len(personas)))

u_d = []
p_d = []

b_time = time.time()

for i in tqdm(range(len(samples_text))):
    u_s = samples_text[i][0]
    p_s = samples_text[i][1]

    u_tmp = torch.stack([model(torch.tensor([tokenizer.encode(u_t)]))[0][0,0] for u_t in u_s], dim = 0)
    p_tmp = torch.stack([model(torch.tensor([tokenizer.encode(p_t)]))[0][0,0] for p_t in p_s], dim = 0)

    u_d.append(u_tmp.data)
    p_d.append(p_tmp.data)

torch.save({'utterance': u_d, 'persona': p_d}, opt.output)

e_time = time.time()

print('time: {}'.format(e_time - b_time))
