import fasttext
from gensim.models import KeyedVectors
import numpy as np
import pandas as pd
import pickle

def load_csv(path, fn):
    df = pd.read_csv(path)
    return df.apply(fn, axis=1).tolist()

def load_embedding(path, node, model='fasttext'):
    if load_embedding.model is None:
        if model == 'fasttext':
            load_embedding.model = fasttext.load_model(path)
        elif model == 'glove':
            load_embedding.model = KeyedVectors.load_word2vec_format(path, binary=False)
        elif model == 'word2vec':
            load_embedding.model = KeyedVectors.load_word2vec_format(path, binary=True)

    if model != 'fasttext':
        if node not in load_embedding.model.vocab:
            # print("Unknown: ", node)
            return np.array(load_embedding.model['unk'])
        else:
            return np.array(load_embedding.model[node])

    return load_embedding.model[node]
load_embedding.model = None
load_embedding.f_model = None

def load_pickle(path):
    with open(path, 'rb') as pickle_file:
        p = pickle.load(pickle_file)

    return p

def write_pickle(path, data):
    with open(path, 'wb') as f:
        pickle.dump(data, f)