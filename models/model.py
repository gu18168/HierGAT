import torch
import torch.nn as nn
import torch.nn.functional as F

from .layer import GraphNgmAttentionLayer as GNAL, GlobalAttentionLayer as GoAL, \
    GraphResAttentionLayer as GRAL, GraphAttentionLayer as GAL, \
    StructAttentionLayer as SAL


class HGAT(nn.Module):
    def __init__(self, n_units, dropout, alpha, nheads, attr_num, sizes=[2, 2, 2]):
        super(HGAT, self).__init__()

        self.dropout = dropout
        self.attr_num = attr_num

        self.attr_inits = nn.ModuleList([
            nn.ModuleList([GoAL(n_units[0], n_units[1], dropout=0, alpha=alpha)
                           for _ in range(nheads)])
            for _ in range(self.attr_num)])
        self.attr_outs = nn.ModuleList([
            GRAL(n_units[1] * nheads, n_units[0], thr=0, dropout=0, alpha=alpha)
            for _ in range(self.attr_num)])
        self.attr_oris = nn.ModuleList([
            nn.ModuleList([GoAL(n_units[0], n_units[1], dropout=0, alpha=alpha)
                           for _ in range(nheads)])
            for _ in range(self.attr_num)])
        self.attr_conts = nn.ModuleList([
            GAL(n_units[1] * nheads, n_units[2], dropout=0, alpha=alpha)
            for _ in range(self.attr_num)])
        self.attr_keys = nn.ModuleList([
            nn.ModuleList([GNAL(n_units[1] * nheads, n_units[3], dropout=0, alpha=alpha, size=size)
                           for size in sizes])
            for _ in range(self.attr_num)])
        self.stru_att = SAL(n_units[4] * attr_num, n_units[4], dropout=0, alpha=alpha)
        self.entity_reses = [GRAL(n_units[4], n_units[4], thr=0.5,
                                  dropout=0, alpha=alpha)
                             for _ in range(1)]

        # Comparion
        en_hidden = n_units[4] * 3
        self.en_fcs = nn.ModuleList([
            torch.nn.Linear(en_hidden, en_hidden),
            torch.nn.Linear(en_hidden, en_hidden),
            torch.nn.Linear(en_hidden, 2)
        ])

        attr_hidden = n_units[4] * 3
        self.key_fcs = nn.ModuleList([
            nn.ModuleList([
                torch.nn.Linear(attr_hidden, attr_hidden),
                torch.nn.Linear(attr_hidden, attr_hidden),
                torch.nn.Linear(attr_hidden, 2)
            ]) for _ in range(self.attr_num)])

    def forward(self, attrs_feature, attrs_adj, attrs_line, p_adj, weights, M):
        attrs_cont_feature = []
        attrs_key_score = []

        for i in range(self.attr_num):
            attr_feature = attrs_feature[i].squeeze()
            attr_adj = attrs_adj[i].squeeze()
            attr_line = attrs_line[i]

            attr_feature = F.dropout(attr_feature,
                                     self.dropout, training=self.training)

            # get entity attribute init embedding, using Global Attention Layer
            attr_init_feature = torch.cat([attr_init(attr_feature, attr_adj, M)
                                           for attr_init in self.attr_inits[i]], dim=1)
            attr_feature = torch.cat([attr_init_feature, attr_feature[M:]], dim=0)

            # broadcast entity attribute init embedding to token embedding
            attr_feature = self.attr_outs[i](attr_feature, attr_adj, attr_feature.size()[0])

            # get entity attribute ori embedding
            attr_ori_feature = torch.cat([attr_ori(attr_feature, attr_adj, M)
                                          for attr_ori in self.attr_oris[i]], dim=1)
            attr_feature = torch.cat([attr_ori_feature, attr_feature[M:]], dim=0)

            # get entity attribute context embedding
            attr_cont_feature = self.attr_conts[i](attr_feature, attr_adj, M)
            attrs_cont_feature.append(attr_cont_feature)

            # get entity attribute keyword embedding & similarity
            attr_key_feature = torch.cat([attr_key(attr_feature, attr_line, M)
                                          for attr_key in self.attr_keys[i]], dim=1)

            l_attr_key = attr_key_feature[:1]
            r_attr_key = attr_key_feature[1:1 + M]
            l_attr_key = l_attr_key.repeat(1, M - 1).view(M - 1, -1)

            lr_attr_key = torch.cat([l_attr_key, r_attr_key,
                                     torch.abs(l_attr_key - r_attr_key)], dim=1)
            lr_attr_key = F.relu(self.key_fcs[i][0](lr_attr_key))
            lr_attr_key = F.relu(self.key_fcs[i][1](lr_attr_key))
            attrs_key_score.append(self.key_fcs[i][2](lr_attr_key))

        attributes_feature = torch.cat(attrs_cont_feature, dim=1)
        entity_feature = self.stru_att(attributes_feature, M, self.attr_num)

        entity_feature = sum([entity_res(entity_feature, p_adj.squeeze(), M)
                              for entity_res in self.entity_reses])

        l_en = entity_feature[:1]
        r_en = entity_feature[1:1 + M]
        l_en = l_en.repeat(1, M - 1).view(M - 1, -1)

        lr_en = torch.cat([l_en, r_en, torch.abs(l_en - r_en)], dim=1)
        lr_en = F.relu(self.en_fcs[0](lr_en))
        lr_en = F.relu(self.en_fcs[1](lr_en))
        en_score = self.en_fcs[2](lr_en)

        scores = [en_score] + attrs_key_score
        score = sum([s * w for (s, w) in zip(scores, weights)])
        return F.log_softmax(score, dim=1)