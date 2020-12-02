from pathlib import Path

import numpy as np
import scipy.sparse as sp
import sklearn
import torch

import pickle
import os
import random

from .block import block_full
from .entity import get_entities
from .graph import get_graph, get_line
from .match import get_idx_match
from .utils import load_embedding, load_pickle

def process(input_files, input_dir, map_file, limit, pattern, map_pattern, id, model_file):
    random.seed(357)

    file_paths = [Path(input_dir, input_file) for input_file in input_files]
    if len(file_paths) != 2:
        raise ValueError("Should provide two csv files to resolution")

    # Get Entities & Idx-Match
    entities, Entity = get_entities(file_paths, pattern)
    idx_match = get_idx_match(Path(input_dir, map_file), entities, map_pattern, id)

    # Blocking
    pos_matches, neg_matches = block_full(entities, idx_match, limit)
    
    with open(Path("output", "pos_matches"), 'wb') as f:
        pickle.dump(pos_matches, f)
    with open(Path("output", "neg_matches"), 'wb') as f:
        pickle.dump(neg_matches, f)

    # Build Attribute Graphs & Lines & Features
    attributes_adjs = []
    attributes_lines = []
    attributes_features = []
    labels = []
    for l_idx, l_entity in enumerate(entities[0]):
        pos_match = pos_matches[l_idx]
        neg_match = neg_matches[l_idx]

        if len(pos_match) == 0 or len(neg_match) < limit:
            continue

        pos_entities = [entities[1][r_idx] for r_idx in pos_match]
        neg_entities = [entities[1][r_idx] for r_idx in neg_match]
        random.shuffle(neg_entities)

        # Ensure 1:M
        for pos in pos_entities:
            index = random.randint(1, limit + 1)
            graph_entities = [l_entity] + neg_entities
            graph_entities.insert(index, pos)

            label = [0 for _ in range(limit)]
            label.insert(index - 1, 1)

            attributes_adj = []
            attributes_line = []
            attributes_feature = []
            for attribute in Entity.attributes.keys():
                # Ignore Id Attribute
                if attribute == id:
                    continue

                (attribute_graph, node_to_idx) = get_graph(graph_entities, attribute, id)

                attribute_adj = []
                for sub_graph in attribute_graph:
                    if len(sub_graph) <= 1:
                        continue

                    l = sub_graph[0]
                    for r in sub_graph[1:]:
                        attribute_adj.append([l, r])

                attribute_line = get_line(graph_entities, node_to_idx, attribute)

                attribute_feature = [0 for _ in range(len(node_to_idx))]
                for node, idx in node_to_idx.items():
                    attribute_feature[idx] = load_embedding(model_file, node, "fasttext")

                attributes_adj.append(attribute_adj)
                attributes_line.append(attribute_line)
                attributes_feature.append(attribute_feature)

            attributes_adjs.append(attributes_adj)
            attributes_lines.append(attributes_line)
            attributes_features.append(attributes_feature)
            labels.append(label)

    return attributes_adjs, attributes_lines, attributes_features, labels