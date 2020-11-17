import torch
import torch.nn as nn

import opts, util
import numpy as np
from tqdm import tqdm
import argparse, sys

from nn_model import DualLinear, SingleLinear
from nn_model import TripletSimLoss, TripletMulLoss, TripletAttLoss

parser = argparse.ArgumentParser(
    description='train.py')
opts.model_opts(parser)
opts.train_opts(parser)
opt = parser.parse_args()
opt.cuda=torch.cuda.is_available() and opt.cuda

print(opt)

np.random.seed(opt.seed)
if opt.cuda:
   torch.cuda.manual_seed(opt.seed)
sys.stdout.flush()


def train():
    train_dataset = torch.load(opt.train_dataset)
    dev_dataset = torch.load(opt.dev_dataset)
    model = SingleLinear(768, 256)
    if opt.cuda:
        train_utterance = [u.cuda() for u in train_dataset['utterance']]
        train_persona = [u.cuda() for u in train_dataset['persona']]

        dev_utterance = [u.cuda() for u in dev_dataset['utterance']]
        dev_persona = [u.cuda() for u in dev_dataset['persona']]

        model = model.cuda()
    else:
        train_utterance = train_dataset['utterance']
        train_persona = train_dataset['persona']

        dev_utterance = dev_dataset['utterance']
        dev_persona = dev_dataset['persona']

    optimizer = torch.optim.Adam(model.parameters(), lr=opt.learning_rate)
    if opt.method == 'mula':
        criterion = TripletMulLoss(type=opt.method, alpha=opt.alpha, cuda=opt.cuda, gamma=opt.gamma)
        train_epoch = train_epoch_negative
    elif opt.method == 'att_soft' or opt.method == 'att_sparse' or opt.method == 'att_sharp':
        criterion = TripletAttLoss(type=opt.method, alpha=opt.alpha, cuda=opt.cuda, gamma=opt.gamma)
        train_epoch = train_epoch_negative
    else: # ap, opt, max_p, max_s, mean
        criterion = TripletSimLoss(type=opt.method, alpha=opt.alpha, cuda=opt.cuda)
        train_epoch = train_epoch_negative

    assert len(train_persona) == len(train_utterance)




    for epoch in range(1, opt.epochs+1):
        print("Epoch {} :".format(epoch))
        train_loss = train_epoch(train_utterance, train_persona, model, criterion, optimizer, mode='train')
        dev_loss = train_epoch(dev_utterance, dev_persona, model, criterion, optimizer, mode='dev')
        print("  --Train {:.4f}  --Dev {:.4f}".format(train_loss, dev_loss))

        # model format: [method]_[leaning_rate]_[alpha]_E[epochs]_L[dev_loss].pt
        if (epoch <= 100 and epoch % opt.save_every == 0) or (epoch > 100 and epoch % (5*opt.save_every) == 0):
            if opt.method == 'att_sharp':
                torch.save(model,
                           opt.save_model + "{}_{}_{}_{}_E{}_L{:.4f}.pt".format(opt.method, opt.learning_rate, opt.alpha, opt.gamma,
                                                                         epoch, dev_loss))
            else:
                torch.save(model,
                           opt.save_model + "{}_{}_{}_E{}_L{:.4f}.pt".format(opt.method, opt.learning_rate, opt.alpha, epoch, dev_loss))

def train_epoch_negative(train_utterance, train_persona, model, criterion, optimizer, mode='train'):
    dataset_size = len(train_utterance)
    batch_loss = 0
    total_loss = 0
    for i in tqdm(range(dataset_size)):
        utt_pos, per_pos = model(train_utterance[i], train_persona[i])
        utt_neg, per_neg = model(train_utterance[np.random.randint(dataset_size)],
                                 train_persona[np.random.randint(dataset_size)])
        batch_loss += criterion(utt_pos, per_pos, utt_neg, per_neg)
        if (i + 1) % opt.batch_size == 0 or i + 1 == dataset_size:
            if mode == 'train':
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()
            total_loss += batch_loss
            batch_loss = 0
    return total_loss / dataset_size

def train_epoch_construct(train_utterance, train_persona, model, criterion, optimizer, mode='train'):
    dataset_size = len(train_utterance)
    batch_loss = 0
    total_loss = 0
    for i in tqdm(range(dataset_size)):
        utt, _utt, per,  _per = model(train_utterance[i], train_persona[i])
        batch_loss += criterion(train_utterance[i], _utt, train_persona[i], _per)
        if (i + 1) % opt.batch_size == 0 or i + 1 == dataset_size:
            if mode == 'train':
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()
            total_loss += batch_loss
            batch_loss = 0
    return total_loss / dataset_size


if __name__ == '__main__':
    torch.manual_seed(opt.seed)
    train()


