import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm

flag = 0
def my_tqdm(i):
    if flag:
        return tqdm(i)
    else:
        return i


class GraphAttentionLayer(nn.Module):

    def __init__(self, in_features, out_features, dropout, alpha, pool=0):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.pool = pool

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)

        self.c1 = nn.Parameter(torch.zeros(size=(out_features, 1)))
        self.c2 = nn.Parameter(torch.zeros(size=(out_features, 1)))

        nn.init.xavier_uniform_(self.c1.data, gain=1.414)
        nn.init.xavier_uniform_(self.c2.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, input, adj, M):
        h = torch.mm(input, self.W)

        h_prime = torch.zeros(M, h.size()[1])
        for x in my_tqdm(range(M)):
            line = adj[x]
            index = np.where(line > 0)[0]

            if len(index) == 0:
                continue

            zh = h[index]
            ind = index[np.where(zh.sum(dim=1) != 0)[0]]

            rh = h[ind]
            L = rh.size()[0]

            e = self.leakyrelu(torch.mm(h[x, :].repeat(L, 1), self.c1) + torch.mm(rh, self.c2)).view(1, L)
            attention = F.softmax(e, dim=1)
            attention = F.dropout(attention, self.dropout, training=self.training)

            if self.pool != 0:
                mask = torch.zeros(L)
                top_idx = attention.topk(1).indices[0][0].item()
                mask[top_idx] = 1

                zero = torch.zeros(1, L)
                attention = torch.where(mask != 0, attention, zero)
                p = L / self.pool
                attention *= p

            h_prime[x] = torch.mm(attention, rh)

        return F.elu(h_prime)


class GlobalAttentionLayer(nn.Module):

    def __init__(self, in_features, out_features, dropout, alpha):
        super(GlobalAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.b = nn.Parameter(torch.zeros(out_features))

        self.a = nn.Parameter(torch.zeros(size=(out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

    def forward(self, input, adj, M):
        h = torch.tanh(torch.mm(input, self.W) + self.b)
        input = torch.mm(input, self.W)

        h_prime = torch.zeros(M, h.size()[1])
        for x in my_tqdm(range(M)):
            line = adj[x]
            index = np.where(line > 0)[0]

            if len(index) == 0:
                continue

            zh = h[index]
            index = index[np.where(zh.sum(dim=1) != 0)[0]]

            rh = h[index]
            ih = input[index]

            L = rh.size()[0]

            e = torch.mm(rh.view(L, self.out_features), self.a).view(1, L)
            attention = F.softmax(e, dim=1)
            attention = F.dropout(attention, self.dropout, training=self.training)

            h_prime[x] = torch.mm(attention, ih)
        return h_prime


class GraphResAttentionLayer(nn.Module):

    def __init__(self, in_features, out_features, thr, dropout, alpha):
        super(GraphResAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.thr = thr
        self.alpha = alpha

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)

        self.a = nn.Parameter(torch.zeros(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, input, adj, M):
        h = torch.mm(input, self.W)

        h_prime = torch.zeros(M, h.size()[1])
        for x in my_tqdm(range(M)):
            line = adj[x]
            index = np.where(line > 0)[0]

            if len(index) == 0:
                continue

            zh = h[index]
            ind = index[np.where(zh.sum(dim=1) != 0)[0]]

            rh = h[ind]
            L = rh.size()[0]

            a_input = torch.cat([h[x, :].repeat(L, 1), rh], dim=1).view(L, 2 * self.out_features)

            e = self.leakyrelu(torch.mm(a_input, self.a)).view(1, L)
            att = F.softmax(e, dim=1)
            att = F.dropout(att, self.dropout, training=self.training)

            top = att.topk(1 if L > 1 else L)
            top_value = top.values[:, -1]

            if top_value > self.thr:
                mask = att >= self.thr

                zero = torch.zeros(1, L)
                attention = torch.where(mask == True, att, zero)
                attention *= L
            else:
                attention = torch.zeros(1, L)

            h_prime[x] = torch.mm(attention, rh)

        return F.elu(h[:M] + h_prime)


class GraphNgmAttentionLayer(nn.Module):

    def __init__(self, in_features, out_features, dropout, alpha, size):
        super(GraphNgmAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.size = size

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)

        self.c1 = nn.Parameter(torch.zeros(size=(out_features, 1)))
        self.c2 = nn.Parameter(torch.zeros(size=(out_features, 1)))

        nn.init.xavier_uniform_(self.c1.data, gain=1.414)
        nn.init.xavier_uniform_(self.c2.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, input, adj, M):
        h = torch.mm(input, self.W)

        h_prime = torch.zeros(M, h.size()[1])
        for x in my_tqdm(range(M)):
            index = adj[x][0].tolist()

            if len(index) == 0:
                continue

            rh = h[index]
            L = rh.size()[0]


            e = self.leakyrelu(torch.mm(h[x, :].repeat(L, 1), self.c1) + torch.mm(rh, self.c2)).view(1, L)
            attention = F.softmax(e, dim=1)
            attention = F.dropout(attention, self.dropout, training=self.training)

            # get mask
            mask = torch.zeros(L)

            if self.size == 1:
                top_idx = attention.topk(1).indices[0][0].item()
                mask[top_idx] = 1
            else:
                # start silding window
                result_atts = -np.ones(L)
                for j in range(self.size - 1, L):
                    slide = torch.sum(attention[0][j - self.size + 1: j + 1])
                    result_atts[j] = slide

                # filter result
                top_idx = np.argmax(result_atts) + 1
                mask[top_idx - self.size:top_idx] = 1

            zeros = torch.zeros_like(attention)
            attention = torch.where(mask != 0, attention, zeros)
            attention *= L / self.size

            h_prime[x] = torch.mm(attention, rh)

        return F.elu(h_prime)


class StructAttentionLayer(nn.Module):

    def __init__(self, in_features, out_features, dropout, alpha):
        super(StructAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha

        self.a = nn.Parameter(torch.zeros(size=(in_features + out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, inputs, M, L):
        h_prime = torch.zeros(M, self.out_features)

        for x in my_tqdm(range(M)):
            inp = inputs[x]
            rh = inp.view(L, -1)

            rh = rh[np.where(rh.sum(dim=1) != 0)[0]]
            l = rh.size()[0]

            a_input = torch.cat([inp.repeat(l, 1), rh], dim=1).view(l, -1)
            e = self.leakyrelu(torch.mm(a_input, self.a)).view(1, l)
            attention = F.softmax(e, dim=1)

            h_prime[x] = torch.mm(attention, rh)

        return F.elu(h_prime)

