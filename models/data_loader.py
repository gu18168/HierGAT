import random

import numpy as np
import scipy.sparse as sp
import sklearn
import torch
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler


class SubgraphDataset(Dataset):

    def __init__(self, attrs_adjs, attrs_lines, nodes_features, labels, limit, seed, shuffle):
        self.entity_num = len(attrs_adjs)
        self.attr_num = len(attrs_adjs[0])

        self.attrs_graphs = [[] for _ in range(self.attr_num)]
        self.attrs_lines = [[] for _ in range(self.attr_num)]
        self.nodes_features = [[] for _ in range(self.attr_num)]

        for attrs_adj, attrs_line, nodes_feature in zip(attrs_adjs, attrs_lines, nodes_features):
            for i, (attr_adj, attr_line, node_feature) in enumerate(zip(attrs_adj, attrs_line, nodes_feature)):
                token_num = len(node_feature)

                attr_adj = np.array(attr_adj)
                attr_graph = sp.coo_matrix((np.ones(attr_adj.shape[0]), (attr_adj[:, 0], attr_adj[:, 1])), shape=(
                    token_num, token_num), dtype=np.float32)

                # attr_graph += sp.eye(attr_graph.shape[0])

                attr_graph = torch.FloatTensor(np.array(attr_graph.todense()))

                self.attrs_graphs[i].append(attr_graph)
                self.attrs_lines[i].append(list(map(lambda x: torch.tensor(x), attr_line)))
                self.nodes_features[i].append(torch.FloatTensor(node_feature))

        self.labels = list(map(lambda label: torch.LongTensor(np.array(label)), labels))
        self.p_adjs = [1 - torch.eye(limit + 2) for _ in range(self.entity_num)]

        if shuffle:
            for i in range(self.attr_num):
                self.attrs_graphs[i], self.attrs_lines[i], self.nodes_features[i] = \
                    sklearn.utils.shuffle(
                        self.attrs_graphs[i], self.attrs_lines[i], self.nodes_features[i],
                        random_state=seed
                    )
            self.labels, self.p_adjs = \
                sklearn.utils.shuffle(
                    self.labels, self.p_adjs,
                    random_state=seed
                )

    def __len__(self):
        return self.entity_num

    def __getitem__(self, idx):
        return [attr_graphs[idx] for attr_graphs in self.attrs_graphs], \
               [attr_lines[idx] for attr_lines in self.attrs_lines], \
               [node_features[idx] for node_features in self.nodes_features], \
               self.labels[idx], self.p_adjs[idx]

    def get_dim(self):
        return self.nodes_features[0][0].size()[1]

    def get_attribute_num(self):
        return self.attr_num


class ChunkSampler(Sampler):
    def __init__(self, num_samples, start=0):
        self.num_samples = num_samples
        self.start = start

    def __iter__(self):
        return iter(range(self.start, self.start + self.num_samples))

    def __len__(self):
        return self.num_samples
