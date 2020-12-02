from argparse import ArgumentParser
from pathlib import Path
import os
import glob
import time
import random
import logging

from models import SubgraphDataset, ChunkSampler, SoftNLLLoss, HGAT
from data import process, load_pickle

import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import classification_report
from tqdm import tqdm

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

parser = ArgumentParser()
# Data Process Arguments

parser.add_argument('-d', '--dir', required=True,
                    dest='input_dir',
                    help="Give input file's dir to process")
parser.add_argument('-l', '--limit', type=int, default=16,
                    dest='limit',
                    help='The number of negative sample')

# Model HyperParameters
parser.add_argument('--no-cuda', action='store_true',
                    default=False,
                    help='Disables CUDA training.')
parser.add_argument('--seed', type=int, default=41,
                    help='Random seed.')
parser.add_argument('--epochs', type=int, default=10,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.0001,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=1e-3,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--dropout', type=float, default=0.2,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--alpha', type=float, default=0.2,
                    help='Alpha for the leaky_relu.')
parser.add_argument('--hidden-units', type=str, default="60,300,100,300",
                    help="Hidden units in each hidden layer, splitted with comma")
parser.add_argument('--weights', type=str, default="1.6,1.0,0.4",
                    help="Weights, splitted with comma")
parser.add_argument('--ratio', type=str, default="1.8,2.5",
                    help="Ratio, splitted with comma")
parser.add_argument('--nb-heads', type=int, default=5,
                    help='Number of head attentions.')
parser.add_argument('--check-point', type=int, default=1,
                    help="Check point")
parser.add_argument('--patience', type=int, default=3,
                    help='Patience')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# Load Data
attrs_adjs = load_pickle(Path(args.input_dir, 'adjs'))
attrs_lines = load_pickle(Path(args.input_dir, 'lines'))
nodes_features = load_pickle(Path(args.input_dir, 'features'))
labels = load_pickle(Path(args.input_dir, 'labels'))

dataset = SubgraphDataset(attrs_adjs, attrs_lines,
                          nodes_features, labels,
                          args.limit, args.seed, True)

N = len(dataset)
feature_dim = dataset.get_dim()
n_units = [feature_dim] + \
          [int(x) for x in args.hidden_units.strip().split(',')]
weights = [float(x) for x in args.weights.strip().split(',')]

train_start, valid_start, test_start = \
    0, int(N * 60 / 100), int(N * (60 + 20) / 100)
train_loader = DataLoader(dataset, batch_size=1,
                          sampler=ChunkSampler(valid_start - train_start, 0))
valid_loader = DataLoader(dataset, batch_size=1,
                          sampler=ChunkSampler(test_start - valid_start, valid_start))
test_loader = DataLoader(dataset, batch_size=1,
                         sampler=ChunkSampler(N - test_start, test_start))

# Get Model & Optimizer & Criterion
model = HGAT(n_units, dropout=args.dropout,
             nheads=args.nb_heads, alpha=args.alpha, attr_num=dataset.get_attribute_num())
optimizer = optim.Adam(model.parameters(), lr=args.lr,
                       weight_decay=args.weight_decay)

ratio = [float(x) for x in args.ratio.strip().split(',')]
criterion = SoftNLLLoss(0.05, torch.Tensor(ratio))


if args.cuda:
    model.cuda()
    attrs_adjs = attrs_adjs.cuda()
    attrs_lines = attrs_lines.cuda()
    nodes_features = node_features.cuda()
    labels = labels.cuda()


def evaluate(thr=None, valid=False):
    model.eval()
    y_true, y_pred, y_score = [], [], []

    eval_loader = valid_loader if valid else test_loader

    loss = 0
    for batch in tqdm(eval_loader):
        attrs_graph, attrs_line, attrs_feature, label, p_adj = batch

        output = model(attrs_feature, attrs_graph, attrs_line, p_adj, weights, args.limit + 2)
        loss_train = criterion(output, label[0])

        loss += loss_train.data.item()
        y_true += label[0].tolist()
        y_pred += output.max(1)[1].data.tolist()
        y_score += output[:, 1].data.tolist()

    y_true = np.array(y_true).reshape(-1, 1)
    y_score = np.array(y_score).reshape(-1, 1)
    y_pred = np.array(y_pred).reshape(-1, 1)

    if thr is not None:
        logger.info("using threshold %.4f", thr)
        y_pred = np.zeros_like(y_score)
        y_pred[y_score > thr] = 1

    logger.info("loss: %.4f All: %d Actual: %d Predict: %d Right: %d",
                loss, len(y_true),
                len(y_true[y_true == 1]), len(y_pred[y_pred == 1]),
                len(y_pred[np.logical_and(y_true == 1, y_pred == 1)]))

    target_names = ['Not Equal', 'Equal']
    print(classification_report(y_true, y_pred, target_names=target_names, digits=3))

    e1_f1 = f1_score(y_true, y_pred)

    if valid:
        precs, recs, thrs = precision_recall_curve(y_true, y_score)
        f1s = 2 * precs * recs / (precs + recs)
        f1s = f1s[:-1]
        thrs = thrs[~np.isnan(f1s)]
        f1s = f1s[~np.isnan(f1s)]
        best_thr = thrs[np.argmax(f1s)]

        e2_f1 = np.max(f1s)
        logger.info("best threshold=%4f, f1=%.4f", best_thr, e2_f1)
        return best_thr, (e1_f1 + e2_f1) / 2
    return None


def train(epoch, train_loader):
    t = time.time()
    model.train()

    loss = 0
    for batch in tqdm(train_loader):
        attrs_graph, attrs_line, nodes_feature, label, p_adj = batch

        optimizer.zero_grad()
        # limit only represents the negative sample num,
        # so we should add one query entity and one positive sample
        output = model(nodes_feature, attrs_graph, attrs_line, p_adj, weights, args.limit + 2)
        loss_train = criterion(output, label[0])

        loss += loss_train.data.item()
        loss_train.backward()
        optimizer.step()

    logger.info('Train Loss Epoch %d: %f, using time: %f',
                epoch + 1, loss, time.time() - t)

    f1 = 0.0
    if (epoch + 1) % args.check_point == 0:
        logger.info("Epoch %d, checkpoint! validation...", epoch + 1)
        _, f1 = evaluate(valid=True)

    return f1


# Train model
t_total = time.time()
f1_values = []
bad_counter = 0
best = 0
best_epoch = 0
for epoch in range(args.epochs):
    f1_values.append(train(epoch, train_loader))

    torch.save(model.state_dict(), '{}.pkl'.format(epoch))
    if f1_values[-1] > best:
        best = f1_values[-1]
        best_epoch = epoch
        bad_counter = 0
    else:
        bad_counter += 1

    if bad_counter > args.patience:
        break

    files = glob.glob('*.pkl')
    for file in files:
        epoch_nb = int(file.split('.')[0])
        if epoch_nb < best_epoch:
            os.remove(file)

files = glob.glob('*.pkl')
for file in files:
    epoch_nb = int(file.split('.')[0])
    if epoch_nb > best_epoch:
        os.remove(file)

logger.info("Optimization Finished!")
logger.info("Total time elapsed: {:.4f}s".format(time.time() - t_total))

# Restore best model
logger.info('Loading {}th epoch'.format(best_epoch + 1))
model.load_state_dict(torch.load('{}.pkl'.format(best_epoch)))

# Evaluate
logger.info("retrieve best threshold...")
best_thr, _ = evaluate(valid=True)

# Test
logger.info("testing...")
evaluate(thr=best_thr)
