import torch, ot
import torch.nn as nn
import pdb
import numpy as np
from scipy.optimize import linear_sum_assignment
from Sparsemax import Sparsemax

class DualLinear(nn.Module):
    def __init__(self, d_emb, d_hid):
        super(DualLinear, self).__init__()
        self.linear_utt = nn.Linear(d_emb, d_hid)
        self.linear_per = nn.Linear(d_emb, d_hid)

    def forward(self, rep_utt, rep_per):
        return self.linear_utt(rep_utt), self.linear_per(rep_per)

class SingleLinear(nn.Module):
    def __init__(self, d_emb, d_hid):
        super(SingleLinear, self).__init__()
        self.linear = nn.Linear(d_emb, d_hid)

    def forward(self, rep_utt, rep_per):
        return self.linear(rep_utt), self.linear(rep_per)


"""
Similarity and Loss
"""

class TripletMulLoss(nn.Module): # approximate version [save]
    def __init__(self, type='mean', alpha=0.5, cuda=True, gamma=1, eps=1e-6):
        super(TripletMulLoss, self).__init__()
        self.alpha = alpha
        self.eps = eps
        self.cuda = cuda
        self.gamma=gamma

    def forward(self, utt_pos, per_pos, utt_neg, per_neg):
        # pdb.set_trace()
        p2p = self.gamma * pairwise_cosine(utt_pos, per_pos, self.eps)
        n2p = self.gamma * pairwise_cosine(utt_neg, per_pos, self.eps)
        p2n = self.gamma * pairwise_cosine(utt_pos, per_neg, self.eps)
        p2p_exp = p2p.exp()
        n2p_exp = n2p.exp()
        p2n_exp = p2n.exp()

        p2p_mi = torch.sum(p2p_exp/p2p_exp.sum() * p2p)
        n2p_mi = torch.sum(n2p_exp/n2p_exp.sum() * n2p)
        p2n_mi = torch.sum(p2n_exp/p2n_exp.sum() * p2n)
        return max(0, self.alpha - p2p_mi + n2p_mi) + max(0, self.alpha - p2p_mi + p2n_mi)


class TripletAttLoss(nn.Module): # attention + KL loss [save]
    def __init__(self, type, alpha=0.2, cuda=True, gamma=1.0, eps=1e-6):
        super(TripletAttLoss, self).__init__()
        self.alpha = alpha
        self.eps = eps
        self.cuda = cuda
        self.type = type
        self.sparsemax = Sparsemax()
        self.gamma = gamma

    def forward(self, utt_pos, per_pos, utt_neg, per_neg):

        p2p = pairwise_cosine(utt_pos, per_pos, self.eps)
        n2p = pairwise_cosine(utt_neg, per_pos, self.eps)
        p2n = pairwise_cosine(utt_pos, per_neg, self.eps)

        if self.type == 'att_sparse':
            p2p_exp = self.sparsemax(p2p.view(-1)).reshape(p2p.size()).data
            n2p_exp = self.sparsemax(n2p.view(-1)).reshape(n2p.size()).data
            p2n_exp = self.sparsemax(p2n.view(-1)).reshape(p2n.size()).data
        elif self.type == 'att_soft':
            p2p_exp = p2p.data.exp()
            n2p_exp = n2p.data.exp()
            p2n_exp = p2n.data.exp()

            p2p_exp = p2p_exp / p2p_exp.sum()
            n2p_exp = n2p_exp / n2p_exp.sum()
            p2n_exp = p2n_exp / p2n_exp.sum()
        elif self.type == 'att_sharp':
            p2p_exp = (self.gamma*p2p).data.exp()
            n2p_exp = (self.gamma*n2p).data.exp()
            p2n_exp = (self.gamma*p2n).data.exp()

            p2p_exp = p2p_exp / p2p_exp.sum()
            n2p_exp = n2p_exp / n2p_exp.sum()
            p2n_exp = p2n_exp / p2n_exp.sum()




        p2p_l = torch.sum(p2p_exp * p2p)
        n2p_l = torch.sum(n2p_exp * n2p)
        p2n_l = torch.sum(p2n_exp * p2n)


        return max(0, self.alpha - p2p_l + n2p_l) + max(0, self.alpha - p2p_l + p2n_l)


class TripletSimLoss(nn.Module):
    def __init__(self, type='mean', alpha=0.5, cuda=True, eps=1e-6):
        super(TripletSimLoss, self).__init__()
        if type == 'mean':
            self.agg_func = torch.mean
        elif type == 'max_p':
            self.agg_func = self.avg_max_p
        elif type == 'max_s':
            self.agg_func = self.avg_max_s
        elif type == 'opt':
            self.agg_func = self.optimal_transport
        elif type == 'ap':
            self.agg_func = self.assignment_problem
        self.alpha = alpha
        self.eps = eps
        self.cuda = cuda

    def avg_max_p(self, sim):
        m, _ = torch.max(sim, dim=0)
        return torch.mean(m)

    def avg_max_s(self, sim):
        m, _ = torch.max(sim, dim=1)
        return torch.mean(m)


    def optimal_transport_weight(self, sim):
        M = sim.data.cpu().numpy()
        num_u, num_p = M.shape

        a = np.ones(num_u)/num_u
        b = np.ones(num_p)/num_p

        weight = torch.FloatTensor(ot.emd(a, b, -M))
        if self.cuda:
            weight = weight.cuda()
        return weight

    def optimal_transport(self, sim):
        weight = self.optimal_transport_weight(sim)
        return torch.sum(weight * sim)

    def assignment_problem_idx(self, sim):
        M = sim.data.cpu().numpy()
        return linear_sum_assignment(-M)


    def assignment_problem(self, sim):
        row_ind, col_ind = self.assignment_problem_idx(sim)
        return torch.sum(sim[row_ind, col_ind])/len(row_ind)


    def forward(self, utt_pos, per_pos, utt_neg, per_neg):
        l1 = self.agg_func(pairwise_cosine(utt_pos, per_pos, self.eps))
        l2 = self.agg_func(pairwise_cosine(utt_neg, per_pos, self.eps))
        l3 = self.agg_func(pairwise_cosine(utt_pos, per_neg, self.eps))

        return max(0, self.alpha-l1+l2) + max(0, self.alpha-l1+l3)



def pairwise_cosine(m1, m2=None, eps=1e-6):
    if m2 is None:
        m2 = m1
    w1 = m1.norm(p=2, dim=1, keepdim=True)
    w2 = m2.norm(p=2, dim=1, keepdim=True)

    return torch.mm(m1, m2.t()) / (w1 * w2.t()).clamp(eps)


